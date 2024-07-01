from typing import Any, Union, List, Optional
from xml.sax.xmlreader import InputSource
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, enable_caching, disable_caching
from linearization import *
from utils import sequence_infilling, build_graph_maps, jointly_infilling
linearization = BaseLinearization()

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
)

def delete_long_sentences(model_inputs, max_length):
    for idx, length in enumerate(model_inputs["length"]):
        if length > max_length:
            for key in model_inputs.keys():
                del model_inputs[key][idx]


def shift_tokens_left(input_ids, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.copy()

    for i in range(len(input_ids)):
        shifted_input_ids[i] = np.roll(input_ids[i], -1)
        shifted_input_ids[i][-1] = -100

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")

    return shifted_input_ids



class TokenSampler(Sampler):

    def __init__(self, data: Union[Dataset, List], max_tokens) -> None:
        super().__init__(data)
        self.data = data
        self.max_tokens = max_tokens

        self.batches = self.compute_batches()
        print(len(self.batches))


    # compute the number of batches and the indices of each batch
    def compute_batches(self):

        batches = []

        batch = []
        max_length = 0
        length_batches = []
        for idx in range(len(self.data)):

            inputs = self.data[idx]['input_ids']

            if "label" in self.data[idx] or True:
                labels = self.data[idx]['labels']
                cur_length = max(len(inputs), len(labels))
            else:
                cur_length = len(inputs)
            
            cur_tokens = (len(batch) + 1) * max(cur_length, max_length)
            if cur_tokens > self.max_tokens:
                if len(batch) == 0:
                    self.logging.warning("Discarding this sample, you should increase your token batch size "
                                    "(Probably your token batch size is smaller than the maximum length of the model)")
                    continue
                else:
                    batches.append(batch)
                    length_batches.append(len(batch) * max_length)
                    max_length = cur_length
                    batch = [idx]
            else:
                batch.append(idx)
                max_length = max(cur_length, max_length)
        
        if len(batch) > 0:
            batches.append(batch)

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def len(self):
        return len(self.batches)

    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     raise NotImplementedError



class BasePLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)

        self.trainer = trainer
        self.tokenizer = trainer.tokenizer
        self.tokenizer.graphs_ids = {}
        self.tokenizer.graphs_snts = {}
        self.tokenizer.graphs_meta = {}
        self.tokenizer.graphs_graph = {}
        self.tokenizer.graphs_alignment = {}
        self.tokenizer.graphs_pred = {}

        self.tokenizer.new_id = 0
        self.model = trainer.model
        self.conf = conf
        self.lan_tokens = {'en': 'en_XX', 'de': 'de_DE', 'ca': 'ca_XX', 'ar': 'ar_AR', 'el': 'el_EL', 'it': 'it_IT', 'ja': 'ja_XX', 'ko': 'ko_KR', 'hi': 'hi_IN', 'pt': 'pt_XX', 'ru': 'ru_RU', 'pl': 'pl_PL', 'zh': 'zh_CN', 'fr': 'fr_XX', 'vi': 'vi_VN', 'sv':'sv_SE', 'es':'es_XX', 'nl': 'nl_XX', 'uk': 'uk_UA', 'fa':'fa_IR'}
        self.k = 0
        self.total_tokens = 0



        train_paths = []
        for path in glob.glob(str(conf.data.train_file)):
            train_paths.append(path)

        eval_paths = []
        for path in glob.glob(str(conf.data.validation_file)):
            eval_paths.append(path)

        test_paths = []
        for path in glob.glob(str(conf.data.test_file)):
            test_paths.append(path)

        self.datasets = load_dataset(conf.data.dataset_name, data_files={'train': train_paths, 'dev': eval_paths, 'test': test_paths})

        enable_caching()
        # disable_caching()

        self.column_names = self.datasets["train"].column_names
        self.input_column = conf.data.input_column
        self.target_column = conf.data.target_column
        self.max_train_target_length = conf.data.max_train_target_length
        self.max_target_length = conf.data.max_target_length
        self.padding = conf.data.pad_to_max_length
                    
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        self.length_data = len(self.datasets["train"])
        
        self.max_tokens = self.conf.data.max_token_batch
        self.labels_pad_id = -100
        self.samples_per_group = self.conf.data.samples_per_group
        self.shuffle = True


    def prepare_data(self, *args, **kwargs):
        if "train" not in self.datasets:
            raise ValueError("--do_train requires a train dataset")

        self.train_dataset = self.datasets["train"]

        self.total_tokens = 0
        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
            load_from_cache_file=not self.conf.data.overwrite_cache,
            remove_columns=self.column_names,
            cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.train.cache'),
        )
        print('Total tokens: ', self.total_tokens)
        print('Model name: ', self.conf.model.model_name_or_path)
        # exit()

        if self.conf.train.do_eval:
            if "validation" not in self.datasets:
                raise ValueError("--do_eval requires a validation dataset")
            
            
            self.eval_dataset = self.datasets["validation"]
            # self.eval_dataset = self.eval_dataset.shuffle(seed=42)

            if self.conf.data.max_val_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.conf.data.max_val_samples))
            
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.data.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file = not self.conf.data.overwrite_cache,
                cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.val.cache'),
            )
            
            self.eval_dataset = self.eval_dataset.sort("length" , reverse=True)


        if self.conf.train.do_predict:
            if "test" not in self.datasets:
                raise ValueError("--do_predict requires a test dataset")

            self.test_dataset = self.datasets["test"]

            if self.conf.data.max_test_samples is not None:
                self.test_dataset = self.test_dataset.select(range(self.conf.data.max_test_samples))

            self.test_dataset = self.test_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.data.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.data.overwrite_cache,
                cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.test.cache'),
            )


    def prepare_test_data(self, *args, **kwargs):
        self.test_dataset = self.datasets["test"]

        if self.conf.data.max_test_samples is not None:
            self.test_dataset = self.test_dataset.select(range(self.conf.data.max_test_samples))

        self.test_dataset = self.test_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.data.overwrite_cache,
            cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.test.cache'),
        )

        self.test_dataset = self.test_dataset.sort("length", reverse=True)

    def sort_test_data(self, *args, **kwargs):
        self.test_dataset = self.test_dataset.sort("length", reverse=True)


    # def setup(self, stage: Optional[str] = None):
    #     raise NotImplementedError



    def sort_filter_and_shuffle(self, original_dataset, shuffle) -> List:

        if shuffle:
            dataset = sorted(
                original_dataset,
                key=lambda item_: (max(len(item_['labels']), len(item_['input_ids']))) * np.random.normal(1.0, 0.1),
                reverse=False
            )
        else:
            dataset = sorted(
                original_dataset,
                key=lambda item_: max(len(item_['labels']), len(item_['input_ids'])),
                reverse=False
            )

        groups = []
        count_removed = 0

        group = []
        for item in dataset:
            group.append(item)
            if len(group) >= self.samples_per_group:
                groups.append(group)
                group = []

        if len(group) > 0:
            groups.append(group)

        if count_removed > 0:
            self.logging.warning(
                "Original number of samples: {}\n"
                "Number of removed samples with max length at {}: {}".
                format(len(dataset), self.max_tokens, count_removed)
            )

        if shuffle:
            indices = np.random.permutation(len(groups))
        else:
            indices = list(range(len(groups)))

        dataset = []
        for idx in indices:
            dataset += groups[idx]

        return dataset


    def train_dataloader(self, *args, **kwargs) -> DataLoader:  
        print("NUMBER of tokens")

        print(sum([sample["length"] for sample in self.train_dataset ]))

        train_dataset = self.sort_filter_and_shuffle(self.train_dataset, shuffle=self.shuffle)
        
        train_batch_sampler = TokenSampler(train_dataset, max_tokens=self.max_tokens)

        return DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )




    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        eval_dataset = sorted(
            self.eval_dataset,
            key=lambda item: (max(len(item['labels']), len(item['input_ids'])))
        )
        validation_batch_sampler = TokenSampler(eval_dataset, max_tokens=self.max_tokens)
        

        return DataLoader(
            eval_dataset,
            batch_sampler=validation_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        
        print(sum([sample["length"] for sample in self.test_dataset ]))
        test_dataset = sorted(
            self.test_dataset,
            key=lambda item: (max(len(item['labels']), len(item['input_ids'])))
        )
        test_batch_sampler = TokenSampler(test_dataset, max_tokens=self.max_tokens)
        
        return DataLoader(
            test_dataset,
            # batch_size=self.conf.train.eval_batch_size,
            batch_sampler=test_batch_sampler,
            collate_fn=self.data_collator,
            # drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )


    def preprocess_function(self, examples):   
        examples_batch = examples.copy()

        snts_tok = {"input_ids":[], "attention_mask":[]}
        amrs_tok = {"input_ids":[], "attention_mask":[]}

        for idx, _ in enumerate(examples_batch['lang']):
            self.tokenizer.src_lang = self.lan_tokens[examples_batch['lang'][idx]]

            if "t5" in self.conf.model.model_name_or_path:
                snt_tok = self.tokenizer(examples_batch['snt'][idx], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)
            else:
                snt_tok = self.tokenizer(examples_batch['snt'][idx], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)

            snts_tok["input_ids"].append(snt_tok["input_ids"])
            snts_tok["attention_mask"].append(snt_tok["attention_mask"])


            # Setup the tokenizer for targets
            self.tokenizer.tgt_lang = self.lan_tokens[examples_batch['lang'][idx]]
            with self.tokenizer.as_target_tokenizer():
                amr_tok = self.tokenizer(examples_batch["amr"][idx], max_length=self.max_train_target_length+1, padding=self.padding, truncation=True)
                amrs_tok["input_ids"].append(amr_tok["input_ids"])
                amrs_tok["attention_mask"].append(amr_tok["attention_mask"])
                self.total_tokens += len(amr_tok["input_ids"])

        amrs_tok['input_ids'] = [inputs[:-1] if len(inputs) == self.max_train_target_length+1 else inputs for inputs in amrs_tok['input_ids']]
            
        model_inputs = snts_tok
        model_inputs["labels"] = amrs_tok["input_ids"].copy()
        model_inputs["length"] = [max(len(input), len(label))for input, label in zip(model_inputs["input_ids"], model_inputs["labels"])] 

        if not "t5" in self.conf.model.model_name_or_path and not "mbart" in self.conf.model.model_name_or_path:
            model_inputs["labels"] = shift_tokens_left(model_inputs["labels"].copy(), self.tokenizer.pad_token_id, decoder_start_token_id=self.conf.train.decoder_start_token_id)

        ids = []
        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_meta[self.tokenizer.new_id] = examples_batch["metadata"][idx]
            self.tokenizer.graphs_graph[self.tokenizer.new_id] = examples["amr"][idx]

            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids
        return model_inputs

    def preprocess_function1(self, examples):   
        examples_batch = examples.copy()
        snt_lang = {}
        for key in self.column_names:
            if key in self.lan_tokens:
                if examples_batch[key][0]:
                    snt_lang[self.lan_tokens[key]] = examples_batch[key]
            
        if not snt_lang:
            snts = examples_batch["snt"]

            if "t5" in self.conf.model.model_name_or_path:
                snts_tok = self.tokenizer([snt for snt in snts], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)

            else:
                snts_tok = self.tokenizer(snts, max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)

        else:
            self.tokenizer.tgt_lang = "en_XX"
            snts_tok = {"input_ids":[], "attention_mask":[]}
            for lang, snt in snt_lang.items():
                self.tokenizer.src_lang = lang
                snt_tok = self.tokenizer(snt, max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)
                for key in snts_tok:
                    snts_tok[key].extend(snt_tok[key])

       #  snts_tok['input_ids'] = [inputs[:-1] if len(inputs) == self.conf.data.max_source_length+1 else inputs for inputs in snts_tok['input_ids']]

        amrs = examples_batch["amr"]
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            amrs_tok = self.tokenizer(amrs, max_length=self.max_train_target_length+1, padding=self.padding, truncation=True)
            amrs_tok['input_ids'] = [inputs[:-1] if len(inputs) == self.max_train_target_length+1 else inputs for inputs in amrs_tok['input_ids']]
            
        
            # if langs repeat list of amrs for each lang
            if snt_lang:
                all_amrs_tok = {"input_ids":[], "attention_mask":[]}

                for lang, _ in snt_lang.items():
                    all_amrs_tok["input_ids"].extend(amrs_tok["input_ids"])
                    all_amrs_tok["attention_mask"].extend(amrs_tok["attention_mask"])

                amrs_tok = all_amrs_tok

        if self.conf.model.direction == "amr":
            model_inputs = snts_tok
            model_targets = amrs_tok
        else:
            model_inputs = amrs_tok
            model_targets = snts_tok


        if not "t5" in self.conf.model.model_name_or_path and not "mbart" in self.conf.model.model_name_or_path:
            # model_inputs["decoder_input_ids"] = model_targets["input_ids"].copy()
            model_inputs["labels"] = shift_tokens_left(model_targets["input_ids"].copy(), self.tokenizer.pad_token_id, decoder_start_token_id=self.conf.train.decoder_start_token_id)
        else:
            model_inputs["labels"] = model_targets["input_ids"].copy()

        model_inputs["length"] = [len(input) for input in model_inputs["labels"]] 

        # repeated examples_batch id per lang
        if snt_lang:
            examples_batch["id"] = examples_batch["id"] * len(snt_lang)
            examples_batch["metadata"] = examples_batch["metadata"] * len(snt_lang)

        # add ids (string) to the model inputs
        ids = []

        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_meta[self.tokenizer.new_id] = examples_batch["metadata"][idx]
            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids

        # delete longest lenght
        # delete_long_sentences(model_inputs, max_target_length)

        return model_inputs

class AlignmentPLDataModule(BasePLDataModule):

    def preprocess_function(self, examples):   
        examples_batch = examples.copy()

        snts_tok = {"input_ids":[], "attention_mask":[]}
        amrs_tok = {"input_ids":[], "attention_mask":[]}

        for idx, _ in enumerate(examples_batch['lang']):
            self.tokenizer.src_lang = self.lan_tokens[examples_batch['lang'][idx]]
            snt_tok = self.tokenizer(examples_batch['snt'][idx], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)
            snts_tok["input_ids"].append(snt_tok["input_ids"])
            snts_tok["attention_mask"].append(snt_tok["attention_mask"])


            # Setup the tokenizer for targets
            self.tokenizer.tgt_lang = self.lan_tokens[examples_batch['lang'][idx]]
            with self.tokenizer.as_target_tokenizer():
                amr_tok = self.tokenizer(examples_batch["amr"][idx], max_length=self.max_train_target_length+1, padding=self.padding, truncation=True)
                amrs_tok["input_ids"].append(amr_tok["input_ids"])
                amrs_tok["attention_mask"].append(amr_tok["attention_mask"])
                self.total_tokens += len(amr_tok["input_ids"])
            
            # if examples_batch["id"][idx] == "PROXY_CNA_ENG_20080408_0041.15":
                
        amrs_tok['input_ids'] = [inputs[:-1] if len(inputs) == self.max_train_target_length+1 else inputs for inputs in amrs_tok['input_ids']]
            
        model_inputs = snts_tok
        model_inputs["labels"] = amrs_tok["input_ids"].copy()
        model_inputs["length"] = [len(input) for input in model_inputs["labels"]] 

        if not "t5" in self.conf.model.model_name_or_path and not "mbart" in self.conf.model.model_name_or_path:
            model_inputs["labels"] = shift_tokens_left(model_inputs["labels"].copy(), self.tokenizer.pad_token_id, decoder_start_token_id=self.conf.train.decoder_start_token_id)


        # add ids (string) to the model inputs
        ids = []
        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_meta[self.tokenizer.new_id] = examples_batch["metadata"][idx]
            self.tokenizer.graphs_graph[self.tokenizer.new_id] = examples["amr"][idx]

            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids
        return model_inputs


class SentencesPLDataModule(BasePLDataModule):
    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(conf, trainer, *args, **kwargs)
        self.tokenizer.graphs_doc_ids = {}


    def preprocess_function1(self, examples):   
        examples_batch = examples.copy()

        snts_tok = {"input_ids":[], "attention_mask":[]}
        amrs_tok = {"input_ids":[], "attention_mask":[]}

        for idx, _ in enumerate(examples_batch['lang']):


            self.tokenizer.src_lang = self.lan_tokens[examples_batch['lang'][idx]]
            snt_tok = self.tokenizer(examples_batch['snt'][idx], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)
            snts_tok["input_ids"].append(snt_tok["input_ids"])
            snts_tok["attention_mask"].append(snt_tok["attention_mask"])

            # Setup the tokenizer for targets
            self.tokenizer.tgt_lang = self.lan_tokens[examples_batch['lang'][idx]]
            with self.tokenizer.as_target_tokenizer():
                amr_tok = self.tokenizer(examples_batch["amr"][idx], max_length=self.max_train_target_length+1, padding=self.padding, truncation=True)
                amrs_tok["input_ids"].append(amr_tok["input_ids"])
                amrs_tok["attention_mask"].append(amr_tok["attention_mask"])

        amrs_tok['input_ids'] = [inputs[:-1] if len(inputs) == self.max_train_target_length+1 else inputs for inputs in amrs_tok['input_ids']]
            
        model_inputs = snts_tok
        model_inputs["labels"] = amrs_tok["input_ids"].copy()
        model_inputs["length"] = [len(input) for input in model_inputs["labels"]] 

        # add ids (string) to the model inputs
        ids = []
        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_meta[self.tokenizer.new_id] = examples_batch["metadata"][idx]
            self.tokenizer.graphs_graph[self.tokenizer.new_id] = examples["amr"][idx]

            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids
        return model_inputs


    def preprocess_function(self, examples):   
        examples_batch = examples.copy()

        snts_tok = {"input_ids":[], "attention_mask":[]}
        # self.tokenizer.src_lang = self.lan_tokens[examples_batch['lang'][0]]
        # snts_tok = self.tokenizer(examples_batch['snt'], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)

        if "t5" in self.conf.model.model_name_or_path:
            # snts_tok = self.tokenizer([prompt + snt for snt, prompt in zip(snts, examples_batch["prompt"])], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)   
            snts = []
            for example in examples_batch['snt']:
                snts.append("Parse English to AMR graph: " + example)

            snts_tok = self.tokenizer(snts)
            
        else:
            snts_tok = self.tokenizer(examples_batch['snt'])

        model_inputs = snts_tok
        model_inputs["length"] = [len(input) for input in model_inputs["input_ids"]] 

        delete_long_sentences(model_inputs, self.conf.data.max_source_length)

        # add ids (string) to the model inputs
        ids = []
        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_doc_ids[self.tokenizer.new_id] = examples_batch["doc_id"][idx]
            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids
        return model_inputs
    
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        
        print(sum([sample["length"] for sample in self.test_dataset ]))
        test_dataset = sorted(
            self.test_dataset,
            key=lambda item: len(item['input_ids'])
        )
        test_batch_sampler = TokenSampler(test_dataset, max_tokens=self.max_tokens)
        
        return DataLoader(
            test_dataset,
            # batch_size=self.conf.train.eval_batch_size,
            batch_sampler=test_batch_sampler,
            collate_fn=self.data_collator,
            # drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )



class ConceptsPLDataModule(BasePLDataModule):
    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(conf, trainer, *args, **kwargs)
        self.tokenizer.graphs_doc_ids = {}

    def sort_test_data(self, *args, **kwargs):
        self.test_dataset = self.test_dataset.sort("length", reverse=True)


    def preprocess_function(self, examples):   
        examples_batch = examples.copy()

        snts_tok = {"input_ids":[], "attention_mask":[]}
        snts_tok = self.tokenizer(examples_batch['snt'], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)
        output_tok = self.tokenizer(examples_batch['concepts'], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)

        model_inputs = snts_tok
        model_inputs["length"] = [len(input) for input in model_inputs["input_ids"]] 

        if not "t5" in self.conf.model.model_name_or_path and not "mbart" in self.conf.model.model_name_or_path:
            model_inputs["labels"] = shift_tokens_left(output_tok["input_ids"].copy(), self.tokenizer.pad_token_id, decoder_start_token_id=self.conf.train.decoder_start_token_id)
        else:
            model_inputs["labels"] = output_tok["input_ids"].copy()

        # add ids (string) to the model inputs
        ids = []
        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_graph[self.tokenizer.new_id] = examples["amrs"][idx]
            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids
        return model_inputs


class MultiTasksPLDataModule(BasePLDataModule):

    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(conf, trainer, *args, **kwargs)
        self.tokenizer.concat_token = self.tokenizer.convert_tokens_to_ids(" <g>")



    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        self.aux_train_dataset = self.train_dataset.map(
            self.multitask_preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_train_dataset,
            batch_size=self.conf.train.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        self.aux_eval_dataset = self.eval_dataset.map(
            self.multitask_preprocess_function_eval,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_eval_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        self.aux_test_dataset = self.test_dataset.map(
                self.multitask_preprocess_function_eval,
                batched=True,
                num_proc=self.conf.data.preprocessing_num_workers,
            ) 

        return DataLoader(
            self.aux_test_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )


    def bucket_sentences(self, model_inputs):
        for idx, length in enumerate(model_inputs["length"]):
            if ((self.trainer.current_epoch < self.conf.train.change_dataloader_epoch) and length < self.length_data*0.9) or \
                ((self.trainer.current_epoch >= self.conf.train.change_dataloader_epoch) and not length >=  self.length_data*0.9):
                for key in model_inputs.keys():
                    del model_inputs[key][idx]

 
    def multitask_preprocess_function(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        labels = inputs["labels"]
        # labels = [snt[1:-1] + [self.tokenizer.concat_token] + " " + graph for snt, graph in zip(snt_ids, graph_ids)]

        prob = 0.15 + min(self.trainer.current_epoch, 8.5)*0.1
        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)

        text_no_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt in snt_ids]
        text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(snt_ids, mask_graph_ids)]

        # mask_snt_ids, mask_graph_ids = jointly_infilling(snt_ids, graph_ids, [self.tokenizer.graphs_alignment[snt_id] for snt_id in inputs["ids"]], self.tokenizer.mask_token_id, self.tokenizer, mlm_prob=0.35)
        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)

        masked_text_no_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt in mask_snt_ids]
        masked_text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] +  graph[:-1] for snt, graph in zip(mask_snt_ids, mask_graph_ids)]

        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)

        no_text_graph_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.concat_token] + graph[:-1] for graph in graph_ids.copy()]
        no_text_masked_graph_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.concat_token] + graph[:-1] for graph in mask_graph_ids]

        text_no_graph_attention_mask = [[1]* len(snt) for snt in text_no_graph_ids]
        text_masked_graph_attention_mask = [[1]* len(snt) for snt in text_masked_graph_ids]
        masked_text_no_graph_attention_mask = [[1]* len(snt) for snt in masked_text_no_graph_ids]
        masked_text_masked_graph_attention_mask = [[1]* len(snt) for snt in masked_text_masked_graph_ids]
        no_text_graph_attention_mask = [[1]* len(snt) for snt in no_text_graph_ids]
        no_text_masked_graph_attention_mask = [[1]* len(snt) for snt in no_text_masked_graph_ids]

        inputs_ids = [text_masked_graph_ids] #, masked_text_masked_graph_ids, no_text_masked_graph_ids]
        attention_masks = [text_masked_graph_attention_mask] #, masked_text_masked_graph_attention_mask, no_text_masked_graph_attention_mask]

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": [], "ids": []}

        for input_id, attention_mask in zip(inputs_ids, attention_masks):
            model_inputs["input_ids"].extend(input_id)
            model_inputs["attention_mask"].extend(attention_mask)
            model_inputs["labels"].extend(labels)
            model_inputs["length"].extend([len(input) for input in input_id])
            model_inputs["ids"].extend(inputs["ids"])

        return model_inputs
        
    def multitask_preprocess_function_eval(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.85)
        text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt, graph in zip(snt_ids, mask_graph_ids)]
        text_masked_graph_attention_mask = [[1]* len(snt) for snt in text_masked_graph_ids]

        model_inputs = {"input_ids": text_masked_graph_ids,
                         "attention_mask": text_masked_graph_attention_mask, 
                         "labels": inputs["labels"], 
                         "length": [len(input) for input in inputs["labels"]], 
                         "ids": inputs["ids"]}
        
        return model_inputs

        


class PretrainingPLDataModule(BasePLDataModule):

    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(conf, trainer, *args, **kwargs)
        self.tokenizer.concat_token = self.tokenizer.convert_tokens_to_ids(" <g>")
        self.tokenizer.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id else 32099
        self.tokenizer.bos_token_id = self.tokenizer.bos_token_id  if self.tokenizer.bos_token_id  else self.tokenizer.pad_token_id

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        self.aux_train_dataset = self.train_dataset.map(
            self.multitask_preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_train_dataset,
            batch_size=self.conf.train.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:        
        self.aux_eval_dataset = self.eval_dataset.map(
            self.multitask_preprocess_function_eval,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_eval_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )

    def multitask_preprocess_function(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        labels = inputs["labels"]
        # labels = [snt[1:-1] + [self.tokenizer.concat_token] + " " + graph for snt, graph in zip(snt_ids, graph_ids)]
        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)
        
        prob = 0.15 + min(int(self.trainer.current_epoch)*2, 8.5)*0.1
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)

        text_no_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt in snt_ids]

        # text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(snt_ids, mask_graph_ids)]
        text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(snt_ids, mask_graph_ids)]

        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)

        masked_text_no_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt in mask_snt_ids]
        masked_text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(mask_snt_ids, mask_graph_ids)]

        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)
        
        no_text_graph_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.concat_token] + graph[:-1] for graph in graph_ids.copy()]
        # no_text_masked_graph_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.concat_token] + graph[:-1] for graph in mask_graph_ids]
        no_text_masked_graph_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.concat_token] + graph[:-1] for graph in mask_graph_ids]

        text_no_graph_attention_mask = [[1]* len(snt) for snt in text_no_graph_ids]
        text_masked_graph_attention_mask = [[1]* len(snt) for snt in text_masked_graph_ids]
        masked_text_no_graph_attention_mask = [[1]* len(snt) for snt in masked_text_no_graph_ids]
        masked_text_masked_graph_attention_mask = [[1]* len(snt) for snt in masked_text_masked_graph_ids]
        no_text_graph_attention_mask = [[1]* len(snt) for snt in no_text_graph_ids]
        no_text_masked_graph_attention_mask = [[1]* len(snt) for snt in no_text_masked_graph_ids]
        
        inputs_ids = [text_masked_graph_ids, masked_text_masked_graph_ids, no_text_masked_graph_ids]
        attention_masks = [text_masked_graph_attention_mask, masked_text_masked_graph_attention_mask, no_text_masked_graph_attention_mask]

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": [], "ids": []}

        for input_id, attention_mask in zip(inputs_ids, attention_masks):
            for idx, input in enumerate(input_id):
                if len(input) < self.conf.data.max_source_length:  
                    model_inputs["input_ids"].append(input)
                    model_inputs["attention_mask"].append(attention_mask[idx])
                    model_inputs["labels"].append(labels[idx])
                    model_inputs["length"].append(len(input))
                    model_inputs["ids"].append(inputs["ids"][idx])
        return model_inputs


    def multitask_preprocess_function_eval(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=1.0)

        text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt, graph in zip(snt_ids, mask_graph_ids)]
        text_masked_graph_attention_mask = [[1]* len(snt) for snt in text_masked_graph_ids]
        
        model_inputs = {"input_ids": text_masked_graph_ids, 
                        "attention_mask": text_masked_graph_attention_mask, 
                        "labels": inputs["labels"], 
                        "length": [len(input) for input in inputs["labels"]], 
                        "ids": inputs["ids"]
            }

        return model_inputs



class BiBLPLDataModule(BasePLDataModule):
    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(conf, trainer, *args, **kwargs)
        self.tokenizer.concat_token = self.tokenizer.convert_tokens_to_ids(" <g>")
        self.tokenizer.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id else 32099
        self.tokenizer.bos_token_id = self.tokenizer.bos_token_id  if self.tokenizer.bos_token_id  else self.tokenizer.pad_token_id

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        self.aux_train_dataset = self.train_dataset.map(
            self.BiBL_preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_train_dataset,
            batch_size=self.conf.train.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:        
        self.aux_eval_dataset = self.eval_dataset.map(
            self.BiBL_preprocess_function_eval,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_eval_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )

    def BiBL_preprocess_function(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        labels = inputs["labels"]
        # labels = [snt[1:-1] + [self.tokenizer.concat_token] + " " + graph for snt, graph in zip(snt_ids, graph_ids)]
        
        text_no_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt in snt_ids]

        prob = 0.01 + min(int(self.trainer.current_epoch)*0.005, 0.14)
        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        masked_text_no_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt in snt_ids]

        prob = 0.1 + min(int(self.trainer.current_epoch)*0.02, 0.75)
        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        masked_text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(mask_snt_ids, mask_graph_ids)]

        text_no_graph_attention_mask = [[1]* len(snt) for snt in text_no_graph_ids]
        masked_text_no_graph_attention_mask = [[1]* len(snt) for snt in masked_text_no_graph_ids]
        masked_text_masked_graph_attention_mask = [[1]* len(snt) for snt in masked_text_masked_graph_ids]

        inputs_ids = [] #, masked_text_no_graph_ids, masked_text_masked_graph_ids]
        attention_masks = [] #, masked_text_no_graph_attention_mask, masked_text_masked_graph_attention_mask]

        if random.random() < 0.3:
            inputs_ids.append(masked_text_no_graph_ids)
            attention_masks.append(masked_text_no_graph_attention_mask)
        else:
            inputs_ids.append(text_no_graph_ids)
            attention_masks.append(text_no_graph_attention_mask)

        # elif random.random() < 0.5:
        #     inputs_ids.append(masked_text_masked_graph_ids)
        #     attention_masks.append(masked_text_masked_graph_attention_mask)

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": [], "ids": []}

        for input_id, attention_mask in zip(inputs_ids, attention_masks):
            for idx, input in enumerate(input_id):
                if len(input) < self.conf.data.max_source_length:  
                    model_inputs["input_ids"].append(input)
                    model_inputs["attention_mask"].append(attention_mask[idx])
                    model_inputs["labels"].append(labels[idx])
                    model_inputs["length"].append(len(input))
                    model_inputs["ids"].append(inputs["ids"][idx])
        
        return model_inputs


    def BiBL_preprocess_function_eval(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=1.0)

        text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt, graph in zip(snt_ids, mask_graph_ids)]
        text_masked_graph_attention_mask = [[1]* len(snt) for snt in text_masked_graph_ids]
        
        model_inputs = {"input_ids": text_masked_graph_ids, 
                        "attention_mask": text_masked_graph_attention_mask, 
                        "labels": inputs["labels"], 
                        "length": [len(input) for input in inputs["labels"]], 
                        "ids": inputs["ids"]
            }

        return model_inputs



class PretrainingEnsemblePLDataModule(PretrainingPLDataModule):

    def multitask_preprocess_function(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        labels = inputs["labels"]


        # text + masked graph
        prob = 0.15 + min(int(self.trainer.current_epoch)*1, 7)*0.1
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(snt_ids, mask_graph_ids)]
        text_masked_graph_attention_mask = [[1]* len(snt) for snt in text_masked_graph_ids]

        # masked text + masked graph
        prob = 0.35 
        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        masked_text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(mask_snt_ids, mask_graph_ids)]
        masked_text_masked_graph_attention_mask = [[1]* len(snt) for snt in masked_text_masked_graph_ids]

        # no text + masked graph
        prob = 0.35 
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        no_text_masked_graph_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.concat_token] + graph[:-1] for graph in mask_graph_ids]
        no_text_masked_graph_attention_mask = [[1]* len(snt) for snt in no_text_masked_graph_ids]
        

        # masked text + masked graphs
        prob = 0.35 
        mask_snt_ids = sequence_infilling(snt_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        prob = 0.55 
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        masked_text_masked_graphs_ids = [snt[:-1] + [self.tokenizer.concat_token] + graph[:-1] for snt, graph in zip(mask_snt_ids, mask_graph_ids)]

        masked_graphs_total = 1
        while len(input) < self.conf.data.max_source_length and masked_graphs_total < 5:  
            masked_graphs_total += 1
            mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)
            masked_text_masked_graphs_ids = [graph[:-1] + [self.tokenizer.concat_token] + masked[:-1] for graph, masked in zip(no_text_masked_graph_ids, mask_graph_ids)]
        
        masked_text_masked_graphs_attention_mask = [[1]* len(snt) for snt in masked_text_masked_graphs_ids]
        
        # text + masked graphs
        prob = 0.35
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=prob)
        no_text_masked_graphs_ids = [[self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.concat_token] + graph[:-1] for graph in mask_graph_ids]

        masked_graphs_total = 1
        while len(input) < self.conf.data.max_source_length and masked_graphs_total < 5:  
            masked_graphs_total += 1
            mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=0.35)
            no_text_masked_graphs_ids = [graph[:-1] + [self.tokenizer.concat_token] + masked[:-1] for graph, masked in zip(no_text_masked_graph_ids, mask_graph_ids)]
        
        no_text_masked_graphs_attention_mask = [[1]* len(snt) for snt in no_text_masked_graphs_ids]
        

        # create input
        inputs_ids = [text_masked_graph_ids, masked_text_masked_graph_ids, no_text_masked_graph_ids, masked_text_masked_graphs_ids, no_text_masked_graphs_ids]
        attention_masks = [text_masked_graph_attention_mask, masked_text_masked_graph_attention_mask, no_text_masked_graph_attention_mask, masked_text_masked_graphs_attention_mask, no_text_masked_graphs_attention_mask]
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": [], "ids": []}

        for input_id, attention_mask in zip(inputs_ids, attention_masks):
            for idx, input in enumerate(input_id):
                if len(input) < self.conf.data.max_source_length:  
                    model_inputs["input_ids"].append(input)
                    model_inputs["attention_mask"].append(attention_mask[idx])
                    model_inputs["labels"].append(labels[idx])
                    model_inputs["length"].append(len(input))
                    model_inputs["ids"].append(inputs["ids"][idx])

        return model_inputs

        



    def multitask_preprocess_function_eval(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        mask_graph_ids = sequence_infilling(graph_ids, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, mlm_prob=1.0)

        text_masked_graph_ids = [snt[:-1] + [self.tokenizer.concat_token, self.tokenizer.mask_token_id, snt[-1]] for snt, graph in zip(snt_ids, mask_graph_ids)]
        text_masked_graph_attention_mask = [[1]* len(snt) for snt in text_masked_graph_ids]
        
        model_inputs = {"input_ids": text_masked_graph_ids, 
                        "attention_mask": text_masked_graph_attention_mask, 
                        "labels": inputs["labels"], 
                        "length": [len(input) for input in inputs["labels"]], 
                        "ids": inputs["ids"]
            }

        return model_inputs



class EnsemblePLDataModule(BasePLDataModule):

    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(conf, trainer, *args, **kwargs)
        self.tokenizer.concat_token = self.tokenizer.convert_tokens_to_ids(" <g>")
        self.tokenizer.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id else 32099
        self.tokenizer.bos_token_id = self.tokenizer.bos_token_id  if self.tokenizer.bos_token_id  else self.tokenizer.pad_token_id

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        self.aux_train_dataset = self.train_dataset.map(
            self.ensemble_preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_train_dataset,
            batch_size=self.conf.train.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:        
        self.aux_eval_dataset = self.eval_dataset.map(
            self.ensemble_preprocess_function_eval,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
        ) 

        return DataLoader(
            self.aux_eval_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        self.aux_test_dataset = self.test_dataset.map(
                self.ensemble_preprocess_function_eval,
                batched=True,
                num_proc=self.conf.data.preprocessing_num_workers,
            ) 

        return DataLoader(
            self.aux_test_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )

    def prepare_test_data(self, *args, **kwargs):
        self.test_dataset = self.datasets["test"]

        if self.conf.data.max_test_samples is not None:
            self.test_dataset = self.test_dataset.select(range(self.conf.data.max_test_samples))

        self.test_dataset = self.test_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.data.overwrite_cache,
            cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.test.cache'),
        )

        self.test_dataset = self.test_dataset.sort("length", reverse=True)


    def sort_test_data(self, *args, **kwargs):
        self.test_dataset = self.test_dataset.sort("length", reverse=True)


    def ensemble_preprocess_function(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        labels = inputs["labels"]
        # pred_graphs = [random.sample(self.tokenizer.graphs_pred[pred_id], random.randint(0, len(self.tokenizer.graphs_pred[pred_id]))) for pred_id in inputs['ids']]
        pred_graphs = [random.sample(self.tokenizer.graphs_pred[pred_id], len(self.tokenizer.graphs_pred[pred_id])) for pred_id in inputs['ids']]
        pred_ids = []

        for idx, preds in enumerate(pred_graphs):
            # aux_pred = snt_ids[idx][:1] if len(preds) > 2 and not random.randint(0,2) else snt_ids[idx][:-1]
            aux_pred = snt_ids[idx][:-1]
            for pred in preds:
                if len(aux_pred + pred) < 2048:
                    aux_pred += [self.tokenizer.concat_token] + pred
            
            aux_pred += [snt_ids[idx][-1]]

            pred_ids.append(aux_pred)


        pred_attention_mask = [[1]* len(snt) for snt in pred_ids]
        
        inputs_ids = [pred_ids]
        attention_masks = [pred_attention_mask]

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": [], "ids": []}

        for input_id, attention_mask in zip(inputs_ids, attention_masks):
            for idx, input in enumerate(input_id):
                if len(input) < self.conf.data.max_source_length:  
                    model_inputs["input_ids"].append(input)
                    model_inputs["attention_mask"].append(attention_mask[idx])
                    model_inputs["labels"].append(labels[idx])
                    model_inputs["length"].append(len(input))
                    model_inputs["ids"].append(inputs["ids"][idx])

        return model_inputs
        
    def ensemble_preprocess_function_eval(self, inputs):
        snt_ids = inputs["input_ids"]
        graph_ids = inputs["labels"]
        labels = inputs["labels"]
        
        pred_graphs = [random.sample(self.tokenizer.graphs_pred[pred_id], len(self.tokenizer.graphs_pred[pred_id])) for pred_id in inputs['ids']]
        # pred_graphs = [self.tokenizer.graphs_pred[pred_id] for pred_id in inputs['ids']]
        pred_ids = []
        for idx, preds in enumerate(pred_graphs):
            aux_pred = snt_ids[idx][:-1]
            for pred in preds:
                if len(aux_pred + pred) < 4062:
                    aux_pred += [self.tokenizer.concat_token] + pred
            
            aux_pred += [snt_ids[idx][-1]]

            pred_ids.append(aux_pred)
        
        pred_attention_mask = [[1]* len(snt) for snt in pred_ids]
        
        inputs_ids = [pred_ids]
        attention_masks = [pred_attention_mask]

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": [], "ids": []}

        for input_id, attention_mask in zip(inputs_ids, attention_masks):
            for idx, input in enumerate(input_id):
                if len(input) < self.conf.data.max_source_length or True:  
                    model_inputs["input_ids"].append(input)
                    model_inputs["attention_mask"].append(attention_mask[idx])
                    model_inputs["labels"].append(labels[idx])
                    model_inputs["length"].append(len(input))
                    model_inputs["ids"].append(inputs["ids"][idx])

        return model_inputs


    def preprocess_function(self, examples):

        snts = examples["snt"]
        amrs = examples["amr"]
        snts_tok = self.tokenizer([snt for snt in snts], max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)
        
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            amrs_tok = self.tokenizer(amrs, max_length=self.max_train_target_length+1, padding=self.padding, truncation=True)
            amrs_tok['input_ids'] = [inputs[:-1] if len(inputs) == self.max_train_target_length+1 else inputs for inputs in amrs_tok['input_ids']]
            # for amr in amrs:
            #     print(self.tokenizer.tokenize(amr))
            # exit()

        if self.conf.model.direction == "amr":
            model_inputs = snts_tok
            model_targets = amrs_tok
        else:
            model_inputs = amrs_tok
            model_targets = snts_tok


        if not "t5" in self.conf.model.model_name_or_path and not "mbart" in self.conf.model.model_name_or_path:
            model_inputs["labels"] = shift_tokens_left(model_targets["input_ids"].copy(), self.tokenizer.pad_token_id, decoder_start_token_id=self.conf.train.decoder_start_token_id)
        else:
            model_inputs["labels"] = model_targets["input_ids"].copy()

        model_inputs["length"] = [len(input) for input in model_inputs["labels"]] 

        ids = []
        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples["id"][idx]
            self.tokenizer.graphs_snts[self.tokenizer.new_id] = examples["snt"][idx]
            self.tokenizer.graphs_meta[self.tokenizer.new_id] = examples["metadata"][idx]

            preds_tok = []

            for pred in examples["amr_preds"][idx].split(" <g> "):
                with self.tokenizer.as_target_tokenizer():
                    pred_tok = self.tokenizer([pred],add_special_tokens=False, max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)["input_ids"][0]

                preds_tok.append(pred_tok)

            self.tokenizer.graphs_pred[self.tokenizer.new_id] = preds_tok.copy()
            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids


        return model_inputs





class GPT2PLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf, trainer: pl.LightningModule, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)

        self.trainer = trainer
        self.tokenizer = trainer.tokenizer
        self.tokenizer.graphs_ids = {}
        self.tokenizer.graphs_snts = {}
        self.tokenizer.graphs_meta = {}
        self.tokenizer.graphs_graph = {}
        self.tokenizer.graphs_alignment = {}
        self.tokenizer.graphs_pred = {}

        self.tokenizer.new_id = 0
        self.model = trainer.model
        self.conf = conf
        self.lan_tokens = {'en': 'en_XX', 'de': 'de_DE', 'ca': 'ca_XX', 'ar': 'ar_AR', 'el': 'el_EL', 'it': 'it_IT', 'ja': 'ja_XX', 'ko': 'ko_KR', 'hi': 'hi_IN', 'pt': 'pt_XX', 'ru': 'ru_RU', 'pl': 'pl_PL', 'zh': 'zh_CN', 'fr': 'fr_XX', 'vi': 'vi_VN', 'sv':'sv_SE', 'es':'es_XX', 'nl': 'nl_XX', 'uk': 'uk_UA', 'fa':'fa_IR'}
        self.k = 0


        train_paths = []
        for path in glob.glob(str(conf.data.train_file)):
            train_paths.append(path)

        eval_paths = []
        for path in glob.glob(str(conf.data.validation_file)):
            eval_paths.append(path)

        test_paths = []
        for path in glob.glob(str(conf.data.test_file)):
            test_paths.append(path)


        self.datasets = load_dataset(conf.data.dataset_name, data_files={'train': train_paths, 'dev': eval_paths, 'test': test_paths})

        enable_caching()
        # disable_caching()

        self.column_names = self.datasets["train"].column_names
        self.input_column = conf.data.input_column
        self.target_column = conf.data.target_column
        self.max_train_target_length = conf.data.max_train_target_length
        self.padding = conf.data.pad_to_max_length
                    
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        self.length_data = len(self.datasets["train"])


    def prepare_data(self, *args, **kwargs):
        if "train" not in self.datasets:
            raise ValueError("--do_train requires a train dataset")

        self.train_dataset = self.datasets["train"]

        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
            load_from_cache_file=not self.conf.data.overwrite_cache,
            remove_columns=self.column_names,
            cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.train.cache'),
        )

        if self.conf.train.do_eval:
            if "validation" not in self.datasets:
                raise ValueError("--do_eval requires a validation dataset")
            
            
            self.eval_dataset = self.datasets["validation"]
            # self.eval_dataset = self.eval_dataset.shuffle(seed=42)

            if self.conf.data.max_val_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.conf.data.max_val_samples))
            
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.data.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file = not self.conf.data.overwrite_cache,
                cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.val.cache'),
            )
            
            self.eval_dataset = self.eval_dataset.sort("length" , reverse=True)


        if self.conf.train.do_predict:
            if "test" not in self.datasets:
                raise ValueError("--do_predict requires a test dataset")

            self.test_dataset = self.datasets["test"]

            if self.conf.data.max_test_samples is not None:
                self.test_dataset = self.test_dataset.select(range(self.conf.data.max_test_samples))

            self.test_dataset = self.test_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.data.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.data.overwrite_cache,
                cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.test.cache'),
            )


    def prepare_test_data(self, *args, **kwargs):
        self.test_dataset = self.datasets["test"]

        if self.conf.data.max_test_samples is not None:
            self.test_dataset = self.test_dataset.select(range(self.conf.data.max_test_samples))

        self.test_dataset = self.test_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.data.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.data.overwrite_cache,
            cache_file_name= self.conf.data.cache_dir + self.conf.data.dataset_name.split('/')[-1].replace('.py', '.test.cache'),
        )

        self.test_dataset = self.test_dataset.sort("length", reverse=True)

    def sort_test_data(self, *args, **kwargs):
        self.test_dataset = self.test_dataset.sort("length", reverse=True)


    # def setup(self, stage: Optional[str] = None):
    #     raise NotImplementedError

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.train.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.train.dataloader_drop_last,
            num_workers=self.conf.train.dataloader_num_workers,
            pin_memory=self.conf.train.dataloader_pin_memory,
        )


    def preprocess_function2(self, examples):   
        examples_batch = examples.copy()

        snts = examples_batch["snt"]

        snts_tok = self.tokenizer(snts, max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)

        amrs = examples_batch["amr"]
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            amrs_tok = self.tokenizer(amrs, max_length=self.max_train_target_length+1, padding=self.padding, truncation=True)
            amrs_tok['input_ids'] = [inputs[:-1] if len(inputs) == self.max_train_target_length+1 else inputs for inputs in amrs_tok['input_ids']]
            
        model_inputs = snts_tok
        model_targets = amrs_tok

        model_inputs["labels"] = model_targets["input_ids"].copy()

        model_inputs["length"] = [len(input) for input in model_inputs["labels"]] 

        # add ids (string) to the model inputs
        ids = []

        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_meta[self.tokenizer.new_id] = examples_batch["metadata"][idx]
            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids

        print(model_inputs.keys())
        print(model_inputs.input_ids.shape)

        return model_inputs


    def preprocess_function(self, examples):
        examples_batch = examples.copy()

        snts = examples_batch["snt"]
        amrs = examples_batch["amr"]
        

        # prepare inputs and labels for GPT2
        model_inputs = self.tokenizer(snts, max_length=self.conf.data.max_source_length, padding=self.padding, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            model_targets = self.tokenizer(amrs, max_length=self.max_train_target_length+1, padding=self.padding, truncation=True)


        model_inputs["labels"] = model_targets["input_ids"].copy()

        model_inputs["length"] = [len(input) for input in model_inputs["labels"]] 

        # add ids (string) to the model inputs
        ids = []

        for idx, _ in enumerate(model_inputs["input_ids"]):
            ids.append(self.tokenizer.new_id)
            self.tokenizer.graphs_ids[self.tokenizer.new_id] = examples_batch["id"][idx]
            self.tokenizer.graphs_meta[self.tokenizer.new_id] = examples_batch["metadata"][idx]
            self.tokenizer.new_id += 1

        model_inputs["ids"] = ids

        return model_inputs

