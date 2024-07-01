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
"""Ensamble AMR3.0"""

from __future__ import absolute_import, division, print_function

import datasets

import logging
from pathlib import Path
import linearization
import penman
from penman.model import Model
from penman.models.noop import NoOpModel
from penman import load, loads
from linearization import *
from itertools import combinations
import random as rd
import glob

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
                    "amr": datasets.Value("string"),
                    "metadata": datasets.Value("string"),
                    "amr_preds": datasets.Value("string"),
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
        gold_graphs_map = {}
        pred_graphs_map = {}
        model = Model() if dereify else NoOpModel()

        linearizer = BaseLinearization()

        for path in filepath:
            path = str(Path(path))
            graphs = load(path, model=model)
            for graph in graphs:
                gold_graphs_map[graph.metadata['id']] = graph
        
        predictions_directory = "/".join(str(Path(filepath[0])).split("/")[:-2]) + "/predictions/*.txt"
        predictions_paths = []
        for path in glob.glob(predictions_directory):
            predictions_paths.append(path)

        for path in predictions_paths:
            path = str(Path(path))
            graphs = load(path, model=model)
            for graph in graphs:
                pred_graphs_map.setdefault(graph.metadata['id'], []).append(graph)

        # pred_graphs_map_sorted = {}
        # for x_id, graphs in pred_graphs_map.items():
        #     pred_graphs_map_sorted[x_id] = [graphs[3,0,1,2]]
# 
        # pred_graphs_map = pred_graphs_map_sorted


        max_lenght=0
        k = 0
        j = 0
        ids = set()
        for x_id, graph in gold_graphs_map.items():
            pred_graphs = pred_graphs_map[x_id]
            x_snt = graph.metadata['snt']
            x_metadata = ""
            for key, value in graph.metadata.items():
                x_metadata += "# ::" + key + " " + value + "\n"


            ids.add(x_id)
            new_graph, gold_graph = linearizer.linearize_extreme(graph, wiki=True)
            if ":ARG1of" in gold_graph and False:
                try:
                    decoded_graph = linearizer.decode_graph_extreme_triples(gold_graph, wiki=True)[0]
                    filter_graph = penman.loads([decoded_graph])[0]
                    exit()
                except:
                    print(x_snt)
                    print(gold_graph)
                    print(decoded_graph)
                    # print(filter_graph)
                    exit()



            try:
                pred_graphs = " <g> ".join([linearizer.linearize_extreme(g, wiki=True)[1] for g in pred_graphs])
            except:
                k += 1
                continue
            
            yield x_id, {
                "id": x_id,
                "snt": x_snt,
                "amr": gold_graph,
                "amr_preds": pred_graphs,
                "metadata": x_metadata
            }


        print("Number of examples removed: ", k)
        print("Number of examples removed because of lenght: ", j)