# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from scipy.sparse.construct import random
from torch_geometric.data import Dataset
from dgl.data import DGLDataset
from sklearn.model_selection import train_test_split
from typing import List
from dgl import DGLGraph
from torch_geometric.data import Data as PYGGraph
import torch
import numpy as np
from typing import Optional, Tuple

from ..wrapper import convert_to_single_emb
from .. import algos
from copy import copy


class GraphormerDGLDataset(Dataset):
    def __init__(self,
        dataset: DGLDataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
    ):
        self.dataset = dataset
        num_data = len(self.dataset)
        self.seed = seed
        if train_idx is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(num_data), test_size=num_data // 10, random_state=seed
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=num_data // 5, random_state=seed
            )
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.__indices__ = None
        self.train_data = self.index_select(train_idx)
        self.valid_data = self.index_select(valid_idx)
        self.test_data = self.index_select(test_idx)

    def index_select(self, indices: List[int]):
        subset = copy(self)
        subset.__indices__ = indices
        subset.train_idx = None
        subset.valid_idx = None
        subset.test_idx = None
        subset.train_data = None
        subset.valid_data = None
        subset.test_data = None
        return subset

    def __extract_edge_and_node_features(
        self, graph_data: DGLGraph
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        def extract_tensor_from_node_or_edge_data(
            feature_dict: dict, num_nodes_or_edges
        ):
            float_feature_list = []
            int_feature_list = []

            def extract_tensor_from_dict(feature: torch.Tensor):
                if feature.dtype == torch.int32 or feature.dtype == torch.long:
                    int_feature_list.append(feature.unsqueeze(1))
                elif feature.dtype == torch.float32 or feature.dtype == torch.float64:
                    float_feature_list.append(feature.unsqueeze(1))

            for feature_or_dict in feature_dict:
                if isinstance(feature_or_dict, torch.Tensor):
                    extract_tensor_from_dict(feature_or_dict)
                elif isinstance(feature_or_dict, dict):
                    for feature in feature_or_dict:
                        extract_tensor_from_dict(feature)
            int_feature_tensor = (
                torch.from_numpy(np.zeros(shape=[num_nodes_or_edges, 1])).long()
                if len(int_feature_list) == 0
                else torch.cat(int_feature_list)
            )
            float_feature_tensor = (
                None if len(float_feature_list) == 0 else torch.cat(float_feature_list)
            )
            return int_feature_tensor, float_feature_tensor

        node_int_feature, node_float_feature = extract_tensor_from_node_or_edge_data(
            graph_data.ndata, graph_data.num_nodes()
        )
        edge_int_feature, edge_float_feature = extract_tensor_from_node_or_edge_data(
            graph_data.edata, graph_data.num_edges()
        )
        return (
            node_int_feature,
            node_float_feature,
            edge_int_feature,
            edge_float_feature,
        )

    def __preprocess_dgl_graph(
        self, graph_data: DGLGraph, y: torch.Tensor, idx: int
    ) -> PYGGraph:
        if not graph_data.is_homogeneous:
            raise ValueError(
                "Heterogeneous DGLGraph is found. Only homogeneous graph is supported."
            )
        N = graph_data.num_nodes()

        (
            node_int_feature,
            node_float_feature,
            edge_int_feature,
            edge_float_feature,
        ) = self.__extract_edge_and_node_features(graph_data)
        edge_index = graph_data.edges()
        attn_edge_type = torch.zeros(
            [N, N, edge_int_feature.shape[1]], dtype=torch.long
        )
        attn_edge_type[
            edge_index[0].long(), edge_index[1].long()
        ] = convert_to_single_emb(edge_int_feature)
        dense_adj = graph_data.adj().to_dense().type(torch.int)
        shortest_path_result, path = algos.floyd_warshall(dense_adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

        pyg_graph = PYGGraph()
        pyg_graph.x = convert_to_single_emb(node_int_feature)
        pyg_graph.adj = dense_adj
        pyg_graph.attn_bias = attn_bias
        pyg_graph.attn_edge_type = attn_edge_type
        pyg_graph.spatial_pos = spatial_pos
        pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.out_degree = pyg_graph.in_degree
        pyg_graph.edge_input = torch.from_numpy(edge_input).long()
        if y.dim() == 0:
            y = y.unsqueeze(-1)
        pyg_graph.y = y
        pyg_graph.idx = idx

        return pyg_graph

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.__indices__ is not None:
                idx = self.__indices__[idx]
            graph, y = self.dataset[idx]
            return self.__preprocess_dgl_graph(graph, y, idx)
        else:
            raise TypeError("index to a GraphormerDGLDataset can only be an integer.")

    def __len__(self) -> int:
        return len(self.dataset) if self.__indices__ is None else len(self.__indices__)
