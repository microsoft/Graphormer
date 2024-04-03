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
    cell_matrix = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=item.pos.dtype)

    aug_pos = torch.matmul(cell_matrix, item.cell)

    item.lnode = item.natoms

    item.atomic_numbers = torch.cat([item.atomic_numbers, torch.ones(len(cell_matrix))], dim=-1).long()
    item.pos = torch.cat([item.pos, aug_pos], dim=0)

    x = item.atomic_numbers.unsqueeze(-1)
    item.tags = torch.cat([torch.ones(item.natoms, dtype=torch.long) * 2, torch.arange(len(cell_matrix), dtype=torch.long) + 3])
    item.natoms += len(cell_matrix)

    N = x.size(0)
    x = convert_to_single_emb(x)

    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias

    item.radius = torch.max(torch.norm(item.pos.unsqueeze(0) - item.pos.unsqueeze(1), p=2, dim=-1)).to(torch.long)
    assert item.radius < 100

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
