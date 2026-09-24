#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json

import numpy as np
import torch
from ogb.utils.mol import smiles2graph
from torch_geometric.data import Data

import fairseq
import graphormer
import graphormer.models
import graphormer.tasks.graph_prediction
from graphormer.data import algos
from graphormer.data.wrapper import preprocess_item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpus", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    adjacency = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ]
    )
    distances, _ = algos.floyd_warshall(adjacency)
    if distances.tolist() != [[0, 1, 2], [1, 0, 1], [2, 1, 0]]:
        raise RuntimeError("Graphormer shortest-path preprocessing returned bad data")

    graph = smiles2graph("CCO")
    item = preprocess_item(
        Data(
            x=torch.from_numpy(graph["node_feat"]).long(),
            edge_index=torch.from_numpy(graph["edge_index"]).long(),
            edge_attr=torch.from_numpy(graph["edge_feat"]).long(),
            y=torch.tensor([0.0]),
        )
    )
    if tuple(item.spatial_pos.shape) != (3, 3):
        raise RuntimeError("Graphormer molecular preprocessing returned a bad shape")

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required but is not available")
    if args.expected_gpus is not None and gpu_count != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} GPUs, found {gpu_count}")

    print(
        json.dumps(
            {
                "cuda_available": cuda_available,
                "fairseq": fairseq.__version__,
                "gpu_count": gpu_count,
                "graphormer": graphormer.__file__,
                "preprocessed_nodes": item.spatial_pos.size(0),
                "torch": torch.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
