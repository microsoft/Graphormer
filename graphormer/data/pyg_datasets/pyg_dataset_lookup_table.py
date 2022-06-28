# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
from .pcqv2_pyg import PCQv2PYG
from .fsmol import FSmolPYG

import torch.distributed as dist

from vpack import breakpt


class MyFSmolPYG(FSmolPYG):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPCQv2PYG, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPCQv2PYG, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyPCQv2PYG(PCQv2PYG):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPCQv2PYG, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPCQv2PYG, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train")
            valid_set = MyZINC(root=root, split="val")
            test_set = MyZINC(root=root, split="test")
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)
        elif name == "pcqm4mv2_pyg":
            root = "datasets"
            inner_dataset = MyPCQv2PYG(root=root)
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test-dev"]
        elif name == "fsmol":
            root = "datasets"
            train_set = MyFSmolPYG(root=root, split="train")
            valid_set = MyFSmolPYG(root=root, split="val")
            test_set = MyFSmolPYG(root=root, split="test")
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        elif train_idx is not None:
            return GraphormerPYGDataset(
                    inner_dataset,
                    seed,
                    train_idx,
                    valid_idx,
                    test_idx,
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed)
            )
