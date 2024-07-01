# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""AMR3.0"""

from __future__ import absolute_import, division, print_function

import datasets

import logging
from pathlib import Path
import linearization
from penman.model import Model
from penman.models.noop import NoOpModel
from penman import load, loads
from linearization import *

import random as rn

_DESCRIPTION = """\
"""

_Tokens = {'en': 'en_XX', 'de': 'de_DE', 'ca': 'ca_XX', 'ar': 'ar_AR', 'el': 'el_EL', 'it': 'it_IT', 'ja': 'ja_XX', 'ko': 'ko_KR', 'hi': 'hi_IN', 'pt': 'pt_XX', 'ru': 'ru_RU', 'pl': 'pl_PL', 'zh': 'zh_CN', 'fr': 'fr_XX', 'vi': 'vi_VN', 'sv':'sv_SE'}

class AMR3Config(datasets.BuilderConfig):
    """BuilderConfig for AMR 3.0."""

    def __init__(self, **kwargs):
        """BuilderConfig for AMR 3.0.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AMR3Config, self).__init__(**kwargs)


class AMR3(datasets.GeneratorBasedBuilder):
    """Rebel 1.0"""

    BUILDER_CONFIGS = [
        AMR3Config(
            name="amr_text",
            version=datasets.Version("1.0.0", ""),
            description="amr text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "snt": datasets.Value("string"),
                    "lang": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="",
#             citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.config.data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": self.config.data_files["test"]}),
        ]


    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)

        dereify = False
        graphs = []
        model = Model() if dereify else NoOpModel()

        linearizer = BaseLinearization()

        snts_file = []
        for path in filepath:
            path = str(Path(path))
            with open(path, "r") as f:
                lines = f.readlines()
                snts_file.extend(lines)


        ids = set()
        repeated_snt = set()
        j = 0

        for snt in snts_file:
            try:
                id, doc_id, snt_id, snt = snt.split("\t")
            except:
                print(snt)
                exit()


            snt = re.sub(r'<.*?>', '', snt)
            snt = re.sub(r'\(.*?\)', '', snt)
            snt = re.sub(r'\[.*?\]', '', snt)
            snt = snt.strip()

            if ((snt.startswith("'")) or (snt.startswith('"'))) and ((snt.endswith("'")) or (snt.endswith('"'))):
                snt = snt[1:-1]

            if snt.startswith("-"):
                snt = snt[1:].strip()

            if id in ids or snt in repeated_snt:
                continue

            ids.add(id)
            repeated_snt.add(snt)
            
            if snt:
                yield id, {
                    "id": id,
                    "doc_id": doc_id + "_" + snt_id,
                    "snt": snt,
                    "lang": "en_XX"
                }
            else:
                j += 1
        print("Number of graphs with more than 100 tokens: ", j)
