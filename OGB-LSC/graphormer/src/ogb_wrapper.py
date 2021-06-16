# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch_geometric.datasets
from pcq_wrapper import MyPygPCQM4MDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
import pickle

def convert_to_single_emb(x, offset=128):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item, noise=False):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    
    all_rel_pos_3d_with_noise = torch.from_numpy(algos.bin_rel_pos_3d_1(item.all_rel_pos_3d, noise=noise)).long()
    rel_pos_3d_attr = all_rel_pos_3d_with_noise[edge_index[0, :], edge_index[1, :]]
    edge_attr = torch.cat([edge_attr, rel_pos_3d_attr[:, None]], dim=-1)
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float) # with graph token
    
    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    item.all_rel_pos_3d_1 = torch.from_numpy(item.all_rel_pos_3d).float()
    return item

class MyPygPCQM4MDataset2(MyPygPCQM4MDataset):
    def __init__(self, root = 'dataset/mypcq_v4'):
        super().__init__(root=root)
        self.all_rel_pos_3d = pickle.load(open('dataset/all_rel_pos_3d.pkl', 'rb'))

    def download(self):
        super(MyPygPCQM4MDataset2, self).download()

    def process(self):
        super(MyPygPCQM4MDataset2, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            item.all_rel_pos_3d = self.all_rel_pos_3d[self.indices()[idx]]
            # donot add noise to test molecules
            if self.indices()[idx] >= 3426030:
                return preprocess_item(item, noise=False)
            return preprocess_item(item, noise=True)
        else:
            return self.index_select(idx)
