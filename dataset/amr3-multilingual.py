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
"""AMR3.0 multilingual"""

from __future__ import absolute_import, division, print_function
import re
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
                    "snt": datasets.Value("string"),
                    "lang": datasets.Value("string"),
                    "amr": datasets.Value("string"),
                    "metadata": datasets.Value("string"),
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
        graphs_training = []
        graphs_validation = []
        model = Model() if dereify else NoOpModel()

        linearizer = BaseLinearization()


        for path in filepath:
            path = str(Path(path))
            # graphs_training.extend(load(path, model=model))
            if "training" in path:
                graphs_training.extend(load(path, model=model))
            else:
                graphs_validation.extend(load(path, model=model))

        # if len > 10000 select 60000 random graphs
        # if len(graphs_training) > 10000:
        #     graphs_training = rn.sample(graphs_training, 60000)
        

        ids = set()
        j = 0
        k = 0

        for g in graphs_training:
            x_id = g.metadata['id']
            x_snt = g.metadata['snt']
            lang = g.metadata['lang']

            x_id = lang + "." + x_id

            if "failed" in g.metadata or len(g.metadata['snt'].split()) > 100:
                j += 1

            elif x_id not in ids:
                k += 1
                ids.add(x_id)


                url = ""
                if "href" in x_snt:
                    url = x_snt.split('href="')[1].split('"')[0]

                x_snt = re.sub(r'<i.*?>', '', x_snt)
                x_snt = re.sub(r'<a.*?>', '', x_snt)
                x_snt = re.sub(r'<xref.*?>', '', x_snt)
                x_snt = re.sub(r'<sup.*?>', '', x_snt)
                x_snt = re.sub(r'< .*?>', '', x_snt)
                x_snt = re.sub(r'</.*?>', '', x_snt)

                if url:
                    x_snt += "[ URL: " + url + " ]"
                    
                g.metadata['snt'] = x_snt

                x_amr = " " + linearizer.linearize(g)

                x_metadata = ""
                for key, value in g.metadata.items():
                    if key not in  ['preferred', 'amr-annotator']:
                        x_metadata += "# ::" + key + " " + value + "\n"

                yield x_id, {
                    "id": x_id,
                    "snt": x_snt,
                    "lang": lang,
                    "amr": x_amr,
                    "metadata": x_metadata,
                }
        print('\nGrafo borrado: ', j)


        for g in graphs_validation:
            x_id = g.metadata['id']
            x_snt = g.metadata['snt']
            lang = g.metadata['lang']

            x_id = lang + "." + x_id

            if len(g.metadata['snt'].split()) > 100:
                j += 1

            elif x_id not in ids:
                k += 1
                ids.add(x_id)


                url = ""
                if "href" in x_snt:
                    url = x_snt.split('href="')[1].split('"')[0]

                x_snt = re.sub(r'<i.*?>', '', x_snt)
                x_snt = re.sub(r'<a.*?>', '', x_snt)
                x_snt = re.sub(r'<xref.*?>', '', x_snt)
                x_snt = re.sub(r'<sup.*?>', '', x_snt)
                x_snt = re.sub(r'< .*?>', '', x_snt)
                x_snt = re.sub(r'</.*?>', '', x_snt)

                if url:
                    x_snt += "[ URL: " + url + " ]"
                    
                g.metadata['snt'] = x_snt

                x_amr = " " + linearizer.linearize(g)

                x_metadata = ""
                for key, value in g.metadata.items():
                    if key not in  ['preferred', 'amr-annotator']:
                        x_metadata += "# ::" + key + " " + value + "\n"

                yield x_id, {
                    "id": x_id,
                    "snt": x_snt,
                    "lang": lang,
                    "amr": x_amr,
                    "metadata": x_metadata,
                }
        print('\nGrafo borrado: ', j)