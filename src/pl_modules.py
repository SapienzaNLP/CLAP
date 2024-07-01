from asyncio import constants
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch import autocast
from utils import sequence_infilling_batch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import Adafactor
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
# import LongT5Config config
from transformers import LongT5Config
import math
import torch.nn.functional as F

from transformers.optimization import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import get_scheduler


from penman import encode, decode
from penman import loads
import smatch 
from sacrebleu import corpus_bleu
from linearization import *
import constants
import utils
from utils import label_smoothed_nll_loss
from tokenizers import AddedToken
linearization = BaseLinearization()

# import bleu score
from sacrebleu import corpus_bleu


class BasePLModule(pl.LightningModule):

    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(conf)

        if "extreme" in conf.data.dataset_name:
            self.decode = linearization.decode_graph_extreme_triples
        else:
            self.decode = linearization.decode_graph_reduced

        self.step_outputs = []

        self.conf = AutoConfig.from_pretrained(
            conf.model.config_name if conf.model.config_name else conf.model.model_name_or_path,
            decoder_start_token_id = conf.train.decoder_start_token_id,
            early_stopping = False,
            no_repeat_ngram_size = 0,
            output_past = False,
            prefix = " ",
            dropout = conf.train.dropout,
            attention_dropout = conf.train.attention_dropout,
            forced_bos_token_id=None,
        )


        tokenizer_kwargs = {
            # "use_fast": conf.model.use_fast_tokenizer,
            # "model_input_names": ["input_ids", "attention_mask", "masked_text_input_ids", "masked_text_attention_mask", "no_text_input_ids", "no_text_attention_mask"],
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            conf.model.tokenizer_name if conf.model.tokenizer_name else conf.model.model_name_or_path,
            **tokenizer_kwargs
        )
        
        self.tokenizer.init_token = conf.model.init_token if 'init_token' in conf.model else 'Ġ'

        
        # add new tokens
        # self.new_tokens = [self.tokenizer.init_token + token for token in constants.new_tokens]
        self.new_tokens = [AddedToken(' ' + token) for token in constants.new_tokens]
        # self.new_tokens.extend(constants.new_tokens)
        self.new_tokens += [AddedToken(token, single_word=False, lstrip=True, rstrip=False) for token in constants.new_tokens_not_init]
        
        # self.new_tokens += [AddedToken(token, single_word=False, lstrip=True, rstrip=False) for token in constants.bmr_tokens]
        
        self.tokenizer.add_tokens(self.new_tokens)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            conf.model.model_name_or_path,
            config=self.conf,
        )

        # if not conf.finetune:
        self.model.resize_token_embeddings(len(self.tokenizer))

        utils.warmup_embeddings(self.model, self.tokenizer)

        print("Vocav size")
        print(len(self.tokenizer))

        self.log(f'val_smatch', 0.0)

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    def forward(self, inputs, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """ 

        # drop lenght 
        inputs.pop('length', None)
        inputs.pop('ids', None)

        outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
        loss = outputs['loss']
        logits = outputs['logits']

        output_dict = {'loss': loss, 'logits': logits}
        return output_dict


    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # with autocast(device_type="cuda", enabled=True):
        forward_output = self.forward(batch)
            
        self.log("loss", forward_output["loss"])
        return forward_output["loss"]


    def generate_step(self, batch: dict, batch_idx: int, set_name: str) -> None:
        
        # pop ids and length
        leghth = batch.pop('length', None)
        ids = batch.pop('ids', None)

        if set_name == 'test':
            forward_output = {}
            forward_output['loss'] = torch.tensor(0.0)
            forward_output['logits'] = torch.tensor(0.0)
        else:
            with torch.no_grad():
                forward_output = self.forward(batch) # ups no labels no loss
        
        gen_kwargs = {
            "max_length": self.hparams.data.max_target_length,
            "early_stopping": self.hparams.train.early_stopping,
            "decoder_start_token_id": self.hparams.train.decoder_start_token_id,
            "no_repeat_ngram_size": self.hparams.train.no_repeat_ngram_size,
            "length_penalty": self.hparams.train.length_penalty,
            "num_beams": 1,
            "temperature": self.hparams.train.temperature,
            "do_sample": self.hparams.train.do_sample,
            "return_dict_in_generate": True,
        }

        forward_output['loss'] = forward_output['loss'].mean().detach()

        metrics = {}
        metrics[f'{set_name}_loss'] = forward_output['loss']
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}

        if set_name == 'test' or self.current_epoch >= self.hparams.train.skip_epochs:
            generated_dict = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                # attention_mask=batch["attention_mask"], # LORA
                use_cache = True,
                **gen_kwargs,
            )
                
            output_ids = generated_dict.sequences
            input_ids = batch["input_ids"]
            cross_attn = generated_dict.cross_attentions

            # convert -100 in padding token id so that the loss is computed only on the labels
            labels = batch["labels"].clone()
            labels[labels == -100] = self.tokenizer.pad_token_id

            predictions = [self.decode(pred) for pred in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

            # predictions = [self.decode(pred) for pred in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
            outputs['status'] = [pred[1] for pred in predictions]
            outputs['predictions'] = [pred[0] for pred in predictions]
            outputs['labels'] = [self.decode(gold)[0] for gold in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]

            outputs['inputs'] = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            outputs['alignments'] = ["" for _ in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]

            # TODO: fix the alignment 
            if False and self.conf.do_alignment:
                # premute the cross attention matrix to have the same shape as the input
                cross_tensor = utils.permute_cross_attn(cross_attn)

                if "extreme" in self.hparams.data.dataset_name:
                    outputs['alignments'] = utils.extract_alignment_extrem_text(cross_tensor, input_ids, labels, outputs['status'], outputs['ids'], self.tokenizer) 
                elif "mbart" in self.hparams.model.model_name_or_path:
                    outputs['alignments'] = utils.extract_alignment_mbart(cross_tensor, input_ids, labels, outputs['status'], self.tokenizer)
                else:
                    outputs['alignments'] = utils.extract_alignment_bart(cross_tensor, input_ids, output_ids, outputs['status'], self.tokenizer) 
        
        # add ids and length
        outputs['ids'] = ids
        outputs['length'] = leghth

        self.step_outputs.append(outputs)

        return outputs

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        return self.generate_step(batch, batch_idx, 'val')


    def test_step(self, batch: dict, batch_idx: int) -> Any:
        return self.generate_step(batch, batch_idx, 'test')

    def generation_step_end_amr(self, predictions: str, labels: str, snts: str, ids: str, alignments: str, set_name: str) -> None:
        pred_path = self.hparams.data.prediction_path
        gold_path = self.hparams.data.gold_path

        pred_graphs = []
        gold_graphs = []

        for pred, gold, snt, id, align in zip(predictions, labels, snts, ids, alignments):
            pred_graph = loads(pred)[0]
            gold_graph = loads(gold)[0]

            pred_graph.metadata["id"] = str(self.tokenizer.graphs_ids[id])
            pred_graph.metadata["snt"] = snt
            gold_graph.metadata["id"] = self.tokenizer.graphs_ids[id]
            gold_graph.metadata["snt"] = snt
            pred_graph.metadata["alignment"] = align

            pred_graphs.append(pred_graph)
            gold_graphs.append(gold_graph)


        pred_graphs = [encode(g) for g in pred_graphs]
        gold_graphs = [encode(g) for g in gold_graphs]

        with open(pred_path, "w") as f0:
            f0.write("\n\n".join(pred_graphs))
            
        with open(gold_path, "w") as f0:
            f0.write("\n\n".join(gold_graphs))

        try:
            with open(pred_path, "r") as f0, open(gold_path, "r") as f1:
                precision, recall, f1 = next(smatch.score_amr_pairs(f0, f1))
        except:
            precision, recall, f1 = 0.0, 0.0, 0.0

        print(f"\n\nSMATCH: {precision:.4f} {recall:.4f} {f1:.4f}\n\n")
        self.log(f'{set_name}_precision', precision)
        self.log(f'{set_name}_recall', recall)
        self.log(f'{set_name}_smatch', f1)


    def generation_step_end_snt(self, predictions: str, labels: str, set_name: str) -> None:
        pred_path = "/".join(self.hparams.data.train_file.split('/')[:-2]) + "/tmp/pred.txt"
        gold_path = "/".join(self.hparams.data.train_file.split('/')[:-2]) + "/tmp/gold.txt"

        with open(pred_path, "w") as f0:
            f0.write("\n".join(predictions))
            
        with open(gold_path, "w") as f0:
            f0.write("\n".join(labels))

        score = corpus_bleu(predictions, labels)

        self.log(f'{set_name}_blue', score)


    def generation_step_end(self, output: dict, set_name: str) -> None:

        predictions = [pred for batch in output for pred in batch['predictions']]
        labels = [gold for batch in output for gold in batch['labels']]
        input = [input for batch in output for input in batch['inputs']]
        alignments = [input for batch in output for input in batch['alignments']]
        ids = [id for batch in output for id in batch['ids'].detach().cpu().numpy()]

        if self.hparams.model.direction == "amr":
            self.generation_step_end_amr(predictions, labels, input, ids, alignments, set_name)
        else:
            self.generation_step_end_snt(predictions, labels, set_name)


    def on_validation_epoch_end(self) -> Any:
        output = self.step_outputs
        if self.current_epoch < self.hparams.train.skip_epochs:
            self.log(f'val_precision', torch.tensor(0.0))
            self.log(f'val_recall', torch.tensor(0.0))
            self.log(f'val_smatch', torch.tensor(0.0))
            self.step_outputs.clear()
            return {}

        self.generation_step_end(output, 'val')
        self.step_outputs.clear()




    def on_test_epoch_end(self) -> Any:
        output = self.step_outputs
        self.generation_step_end(output, 'test')
        self.step_outputs.clear()

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """

        # Prepare RADAM optimizer
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.train.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
                        "scale_parameter": False,  "relative_step": False,
                        # "betas": self.hparams.train.betas,
                        # "eps": self.hparams.train.eps,
                        "lr": self.hparams.train.lr,
                    }
        
        # optimizer = RAdam(optimizer_grouped_parameters, **optimizer_kwargs)
        optimizer = Adafactor(optimizer_grouped_parameters, **optimizer_kwargs)

        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, self.hparams.train.warmup_steps, self.hparams.train.t_total)
        # lr_scheduler = get_constant_schedule_with_warmup(optimizer, self.hparams.train.warmup_steps)
        lr_scheduler = self.get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.train.warmup_steps, num_training_steps=self.hparams.train.t_total)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]


    def get_inverse_sqrt_schedule_with_warmup(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.
        Args:
            optimizer (:class:`~torch.optim.Optimizer`):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
                The total number of training steps.
            num_cycles (:obj:`float`, `optional`, defaults to 0.5):
                The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
                following a half-cosine).
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.
        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step):
            current_step_aux = current_step # % (num_training_steps/10)
            if current_step_aux < num_warmup_steps:
                return float(current_step_aux) / float(max(1, num_warmup_steps))
            return max(0.0, (num_warmup_steps / current_step_aux)**0.5)

        return LambdaLR(optimizer, lr_lambda, last_epoch)


class SentencesPLModule(BasePLModule):

    def generate_step(self, batch: dict, batch_idx: int, set_name: str) -> None:
        
        # pop ids and length
        leghth = batch.pop('length', None)
        ids = batch.pop('ids', None)

        forward_output = {}
        forward_output['loss'] = torch.tensor(0.0)
        forward_output['logits'] = torch.tensor(0.0)
            
        gen_kwargs = {
            "max_length": self.hparams.data.max_target_length,
            "early_stopping": self.hparams.train.early_stopping,
            "decoder_start_token_id": self.hparams.train.decoder_start_token_id,
            "no_repeat_ngram_size": self.hparams.train.no_repeat_ngram_size,
            "length_penalty": self.hparams.train.length_penalty,
            "num_beams": self.hparams.train.num_beams,
            "temperature": self.hparams.train.temperature,
            "do_sample": self.hparams.train.do_sample,
        }

        forward_output['loss'] = forward_output['loss'].mean().detach()
        outputs = {}

        output_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # attention_mask=batch["attention_mask"], # LORA
            use_cache = True,
            **gen_kwargs,
        )

        input_ids = batch["input_ids"]
        # cross_attn = generated_dict.cross_attentions

        # convert -100 in padding token id so that the loss is computed only on the labels

        predictions = [self.decode(pred) for pred in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


        status = [pred[1] for pred in predictions]
        predictions = [pred[0] for pred in predictions]
        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        pred_graphs = []

        for pred, snt, id, state in zip(predictions, input_text, ids.detach().cpu().numpy(), status):
            if not state:
                continue 
            
            pred_graph = loads(pred)[0]

            pred_graph.metadata["id"] = str(self.tokenizer.graphs_ids[id])
            pred_graph.metadata["doc_id"] = str(self.tokenizer.graphs_doc_ids[id])
            pred_graph.metadata["snt"] = snt.replace("Parse English to AMR graph: ", "")
            pred_graph.metadata["model"] = "T5-large-dfs"

            pred_graphs.append(pred_graph)


        pred_graphs = [encode(g) for g in pred_graphs]

        with open(self.hparams.data.prediction_path, "a") as f0:
            f0.write("\n\n".join(pred_graphs))


        return

    def generation_step_end_amr(self, predictions: str, snts: str, ids: str, set_name: str) -> None:
        pred_path = self.hparams.data.prediction_path
        pred_graphs = []

        for pred, snt, id in zip(predictions, snts, ids):
            pred_graph = loads(pred)[0]

            pred_graph.metadata["id"] = str(self.tokenizer.graphs_ids[id])
            pred_graph.metadata["doc_id"] = str(self.tokenizer.graphs_doc_ids[id])
            pred_graph.metadata["snt"] = snt.replace("Parse English to AMR graph: ", "")
            pred_graph.metadata["lang"] = self.tokenizer.convert_ids_to_tokens(self.hparams.train.decoder_start_token_id)

            pred_graphs.append(pred_graph)


        pred_graphs = [encode(g) for g in pred_graphs]

        with open(pred_path, "w") as f0:
            f0.write("\n\n".join(pred_graphs))

    def generation_step_end(self, output: dict, set_name: str) -> None:
        return
        predictions = [pred for batch in output for pred in batch['predictions']]
        input = [input for batch in output for input in batch['inputs']]
        # alignments = [input for batch in output for input in batch['alignments']]
        ids = [id for batch in output for id in batch['ids'].detach().cpu().numpy()]

        self.generation_step_end_amr(predictions, input, ids, set_name)


class AlignerPLModule(BasePLModule):
    def forward(self, inputs, **kwargs) -> dict:
        # drop lenght 
        inputs.pop('length', None)
        inputs.pop('ids', None)

        outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True, output_attentions=True)

        return outputs
           
    def test_step(self, batch: dict, batch_idx: int) -> Any:
        outputs = {}

        labels = batch["labels"].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        input_ids = batch["input_ids"]

        outputs['ids'] = batch['ids']
        outputs['labels'] = [self.decode(gold)[0] for gold in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]
        outputs['inputs'] = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        outputs['alignments'] = ["" for _ in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]

        # remove pad id from input
        outputs['langs'] = [self.tokenizer.convert_ids_to_tokens([input_ids[i][input_ids[i] != self.tokenizer.pad_token_id][-1]])[0] for i in range(len(input_ids))]

        outputs['status'] = [True for lable in labels]
        with torch.no_grad():
            forward_output = self.forward(batch) # ups no labels no loss

        cross_attn = forward_output.cross_attentions


        cross_tensor = utils.permute_cross_attn_forward(cross_attn, self.hparams.model.model_name_or_path)

        # compute the alignment
        # outputs['alignments'] = utils.extract_alignment_mbart(cross_tensor, input_ids, labels, outputs['status'], self.tokenizer) 
        if "extreme" in self.hparams.data.dataset_name:
            outputs['alignments'] = utils.extract_alignment_extrem_text(cross_tensor, input_ids, labels, outputs['status'], outputs['ids'], self.tokenizer) 
        elif "mbart" in self.hparams.model.model_name_or_path:
            outputs['alignments'] = utils.extract_alignment_mbart(cross_tensor, input_ids, labels, outputs['status'], self.tokenizer)
        else:
            outputs['alignments'] = utils.extract_alignment_bart(cross_tensor, input_ids, labels, outputs['status'], self.tokenizer) 


        self.step_outputs.append(outputs)

        return outputs


    def on_test_epoch_end(self) -> Any:
        output = self.step_outputs
        
        labels = [gold for batch in output for gold in batch['labels']]
        input = [input for batch in output for input in batch['inputs']]
        alignments = [input for batch in output for input in batch['alignments']]
        langs = [input for batch in output for input in batch['langs']]
        ids = [id for batch in output for id in batch['ids'].detach().cpu().numpy()]

        id_labels_map = {}
        for id, label in zip(ids, labels):
            if str(self.tokenizer.graphs_ids[id]) not in id_labels_map:
                id_labels_map[str(self.tokenizer.graphs_ids[id])] = (label, id)

        id_alignment_map = {}
        for id, lang, alignment, snt in zip(ids, langs, alignments, input):
            id_alignment_map.setdefault(str(self.tokenizer.graphs_ids[id]), {})[lang] = (snt, alignment)

        file_path = self.hparams.data.prediction_path
        graphs = []
        

        for id, (graph, int_id) in id_labels_map.items():
            graph = loads(self.decode(self.tokenizer.graphs_graph[int_id])[0])[0]

            metadata = self.tokenizer.graphs_meta[int_id]
            meta = {}
            for line in metadata.split("\n"):
                if line:
                    key = line.strip().split()[1][2:]
                    value = " ".join(line.strip().split()[2:])
                    meta[key] = value
            
            graph.metadata = meta
            
            for lang, (snt, align) in id_alignment_map[id].items():                    
                graph.metadata[f"alignments"] = align

            graphs.append(graph)

        graphs = [encode(g) for g in graphs]

        with open(file_path, "w") as f0:
            f0.write("\n\n".join(graphs))

        self.step_outputs.clear()


class EnsamblePLModule(BasePLModule):

    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__(conf)
        self.save_hyperparameters(conf)

        self.conf = LongT5Config.from_pretrained(
            conf.model.config_name if conf.model.config_name else conf.model.model_name_or_path,
            decoder_start_token_id = conf.train.decoder_start_token_id,
            early_stopping = False,
            no_repeat_ngram_size = 0,
            output_past = False,
            # prefix = " ",
            output_attentions = True,
            dropout = conf.train.dropout,
            attention_dropout = conf.train.attention_dropout,
            forced_bos_token_id=None,
            encoder_attention_type= 'transient-global',
            # local_radius = 20
        )

        tokenizer_kwargs = {
            # "use_fast": conf.model.use_fast_tokenizer,
            # "model_input_names": ["input_ids", "attention_mask", "masked_text_input_ids", "masked_text_attention_mask", "no_text_input_ids", "no_text_attention_mask"],
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            conf.model.tokenizer_name if conf.model.tokenizer_name else conf.model.model_name_or_path,
            **tokenizer_kwargs
        )

        self.tokenizer.init_token = conf.model.init_token if 'init_token' in conf.model else 'Ġ'

        # add new tokens
        self.new_tokens = [AddedToken(' ' + token) for token in constants.new_tokens]
        self.new_tokens += [AddedToken(token, single_word=False, lstrip=True, rstrip=False) for token in constants.new_tokens_not_init]
        
        self.tokenizer.add_tokens(self.new_tokens)


        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            conf.model.model_name_or_path,
            config=self.conf,
            from_flax=True,
            # gradient_checkpointing=True,
        )


        # if not conf.finetune:
        self.model.resize_token_embeddings(len(self.tokenizer))

        utils.warmup_embeddings(self.model, self.tokenizer)

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


class ProbabilityPLModule(BasePLModule):
    def forward(self, inputs, **kwargs) -> dict:
        # drop lenght 
        inputs.pop('length', None)
        ids = inputs.pop('ids', None)
        
        outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)

        return outputs

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        outputs = {}

        # convert -100 in padding token id so that the loss is computed only on the labels
        labels = batch["labels"].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        input_ids = batch["input_ids"]
        lengths = batch["length"]

        outputs['ids'] = batch['ids']
        outputs['labels'] = [self.decode(gold)[0] for gold in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]
        outputs['inputs'] = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        outputs['alignments'] = ["" for _ in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]

        # remove pad id from input
        outputs['langs'] = [self.tokenizer.convert_ids_to_tokens([input_ids[i][input_ids[i] != self.tokenizer.pad_token_id][-1]])[0] for i in range(len(input_ids))]

        outputs['status'] = [True for lable in labels]
        with torch.no_grad():
            forward_output = self.forward(batch) # ups no labels no loss

        scores = []

        for label, logits, length in zip(labels, forward_output["logits"], lengths):
            score = F.cross_entropy(logits[:length], label[:length], ignore_index=self.tokenizer.pad_token_id)
            scores.append(math.exp(score.item()))

        outputs['scores'] = scores

        self.step_outputs.append(outputs)

        return outputs

    def on_test_epoch_end(self) -> Any:
        output = self.step_outputs
        
        labels = [gold for batch in output for gold in batch['labels']]
        input = [input for batch in output for input in batch['inputs']]
        scores = [input for batch in output for input in batch['scores']]
        langs = [input for batch in output for input in batch['langs']]
        ids = [id for batch in output for id in batch['ids'].detach().cpu().numpy()]

        id_labels_map = {}
        for id, label in zip(ids, labels):
            if str(self.tokenizer.graphs_ids[id]) not in id_labels_map:
                id_labels_map[str(self.tokenizer.graphs_ids[id])] = (label, id)

        id_scores_map = {}
        for id, lang, score, snt in zip(ids, langs, scores, input):
            id_scores_map.setdefault(str(self.tokenizer.graphs_ids[id]), {})[lang] = (snt, score)

        file_path = self.hparams.data.prediction_path
        graphs = []
        


        for id, (graph, int_id) in id_labels_map.items():
            graph = loads(graph)[0]

            
            metadata = self.tokenizer.graphs_meta[int_id]
            for line in metadata.split("\n"):
                if line:
                    key = line.strip().split()[1][2:]
                    value = " ".join(line.strip().split()[2:])
                    graph.metadata[key] = value

            graph.metadata["id"] = str(self.tokenizer.graphs_ids[int_id])
            
            for lang, (snt, score) in id_scores_map[id].items():                    
                graph.metadata[f"scores"] = str(score)

            graphs.append(graph)

        graphs = [encode(g) for g in graphs]

        with open(file_path, "w") as f0:
            f0.write("\n\n".join(graphs))


        self.step_outputs.clear()

