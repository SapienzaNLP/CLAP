import re
import logging
from pathlib import Path

import datasets
from penman import load
from linearization import BaseLinearization

# Global variable for tokens in different languages
LANGUAGE_TOKENS = {
    'en': 'en_XX', 'de': 'de_DE', 'ca': 'ca_XX', 'ar': 'ar_AR', 'el': 'el_EL',
    'it': 'it_IT', 'ja': 'ja_XX', 'ko': 'ko_KR', 'hi': 'hi_IN', 'pt': 'pt_XX',
    'ru': 'ru_RU', 'pl': 'pl_PL', 'zh': 'zh_CN', 'fr': 'fr_XX', 'vi': 'vi_VN', 'sv': 'sv_SE'
}

class AMR3Config(datasets.BuilderConfig):
    """ Configuration class for AMR 3.0 dataset builder. """
    def __init__(self, **kwargs):
        super(AMR3Config, self).__init__(**kwargs)

class AMR3(datasets.GeneratorBasedBuilder):
    """ Dataset builder class for AMR 3.0. """
    BUILDER_CONFIGS = [
        AMR3Config(
            name="amr_text",
            version=datasets.Version("1.0.0"),
            description="AMR dataset version 3.0 text",
        ),
    ]

    def _info(self):
        """ Returns the dataset information. """
        return datasets.DatasetInfo(
            description="AMR dataset version 3.0",
            features=datasets.Features({
                "id": datasets.Value("string"),
                "snt": datasets.Value("string"),
                "amr": datasets.Value("string"),
                "metadata": datasets.Value("string"),
                "lang": datasets.Value("string"),
            }),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """ Returns SplitGenerators. """
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.config.data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": self.config.data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """ Yields examples from the filepath. """
        logging.info("Generating examples from = %s", filepath)
        linearizer = BaseLinearization()
        ids = set()
        for path in filepath:
            path = str(Path(path))
            graphs = load(path)

            for graph in graphs:
                snt = self.process_sentence(graph.metadata.get('snt', '').replace('"', "'"))
                id = graph.metadata.get('id', graph.metadata.get('nsent', ''))
                metadata = self.format_metadata(graph.metadata)

                if id in ids:
                    continue
                
                ids.add(id)

                try:
                    _, amr = linearizer.linearize_extreme(graph)
                except Exception as e:
                    logging.warning(f"Failed to linearize graph {id}: {str(e)}")
                    continue
                import penman

                yield id, {
                    "id": id,
                    "snt": snt,
                    "amr": amr,
                    "metadata": metadata,
                    "lang": "en"  # assuming language is English
                }

    def process_sentence(self, sentence):
        """ Cleans up sentences to remove HTML tags and add placeholders for URLs. """
        sentence = re.sub(r'<a href="([^"]+)">', r'<a href="\1"> URL_LINK ', sentence)
        sentence = re.sub(r'<[^>]+>', '', sentence)  # Remove all HTML tags
        return sentence.strip()

    def format_metadata(self, metadata):
        """ Formats the metadata into a string. """
        return "\n".join(f"# ::{key} {value}" for key, value in metadata.items())
