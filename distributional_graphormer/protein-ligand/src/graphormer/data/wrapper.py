# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np

from . import algos


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    num_edge = item.num_edge
    #edge_attr [lig_ei, pro_ei, lig_sei,  pro_sei, lig_pock_ei]
    #num_edge [lig_ei, pro_ei, lig_pock_ei, lig_sei,  pro_sei]
    num_original_edge = torch.sum(num_edge[:2]) # lig_ei + pro_ei
    num_pro_spatial_edge = num_edge[-1] # pro_sei
    num_lig_pro_edge = num_edge[2] # lig_pock_ei
    # keep lig_ei, pro_ei, pro_sei
    # TODO: change pro_sei to crystal pro_sei
    if num_lig_pro_edge > 0:
        edge_attr = torch.cat([
                edge_attr[:num_original_edge, :],
                edge_attr[-(num_pro_spatial_edge + num_lig_pro_edge) : -num_lig_pro_edge, : ],
            ],axis=0,)
        edge_index = torch.cat([
                edge_index[:, :num_original_edge],
                edge_index[:, -(num_pro_spatial_edge + num_lig_pro_edge) : -num_lig_pro_edge],
            ],axis=1,)
    else:
        edge_attr = torch.cat([
                edge_attr[:num_original_edge, :],
                edge_attr[-(num_pro_spatial_edge + num_lig_pro_edge) : , :],
            ],axis=0,)
        edge_index = torch.cat([
                edge_index[:, :num_original_edge],
                edge_index[:, -(num_pro_spatial_edge + num_lig_pro_edge) :],
            ],axis=1,)
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    num_lig_node = item.num_node[0]
    shortest_path_result[:num_lig_node, num_lig_node:] = 511
    shortest_path_result[num_lig_node:, :num_lig_node] = 511
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


def preprocess_pred_info(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    return {
        "x": x,
        "adj": adj,
        "edge_type": attn_edge_type,
        "spatial_pos": torch.tensor(shortest_path_result),
    }
