# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sklearn.model_selection import train_test_split
import torch
import numpy as np

from ..wrapper import preprocess_item
from .. import algos
from ..pyg_datasets import GraphormerPYGDataset

from ogb.utils.mol import smiles2graph


class GraphormerSMILESDataset(GraphormerPYGDataset):
    def __init__(
        self,
        dataset: str,
        num_class: int,
        max_node: int,
        multi_hop_max_dist: int,
        spatial_pos_max: int,
    ):
        self.dataset = np.genfromtxt(dataset, delimiter=",", dtype=str)
        num_data = len(self.dataset)
        self.num_class = num_class
        self.__get_graph_metainfo(max_node, multi_hop_max_dist, spatial_pos_max)
        train_valid_idx, test_idx = train_test_split(num_data // 10)
        train_idx, valid_idx = train_test_split(train_valid_idx, num_data // 5)
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.__indices__ = None
        self.train_data = self.index_select(train_idx)
        self.valid_data = self.index_select(valid_idx)
        self.test_data = self.index_select(test_idx)

    def __get_graph_metainfo(
        self, max_node: int, multi_hop_max_dist: int, spatial_pos_max: int
    ):
        self.max_node = min(
            max_node,
            torch.max(self.dataset[i][0].num_nodes() for i in range(len(self.dataset))),
        )
        max_dist = 0
        for i in range(len(self.dataset)):
            pyg_graph = smiles2graph(self.dataset[i])
            dense_adj = pyg_graph.adj().to_dense().type(torch.int)
            shortest_path_result, _ = algos.floyd_warshall(dense_adj.numpy())
            max_dist = max(max_dist, np.amax(shortest_path_result))
        self.multi_hop_max_dist = min(multi_hop_max_dist, max_dist)
        self.spatial_pos_max = min(spatial_pos_max, max_dist)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = smiles2graph(self.dataset[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")
