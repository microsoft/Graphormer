# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from dgl.data import (
    QM7bDataset,
    QM9Dataset,
    QM9EdgeDataset,
    MiniGCDataset,
    TUDataset,
    GINDataset,
    FakeNewsDataset,
)
from dgl.data import DGLDataset
from .dgl_dataset import GraphormerDGLDataset

import torch.distributed as dist

class MyQM7bDataset(QM7bDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7bDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

class MyQM9Dataset(QM9Dataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9Dataset, self).download()
        if dist.is_initialized():
            dist.barrier()

class MyQM9EdgeDataset(QM9EdgeDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9EdgeDataset, self).download()
        if dist.is_initialized():
            dist.barrier()


class MyMiniGCDataset(MiniGCDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMiniGCDataset, self).download()
        if dist.is_initialized():
            dist.barrier()


class MyTUDataset(TUDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyTUDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

class MyGINDataset(GINDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyGINDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

class MyFakeNewsDataset(FakeNewsDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyFakeNewsDataset, self).download()
        if dist.is_initialized():
            dist.barrier()


class DGLDatasetLookupTable:
    @staticmethod
    def GetDGLDataset(dataset_name: str, seed: int) -> Optional[DGLDataset]:
        params = dataset_name.split(":")[-1].split(",")
        inner_dataset = None

        if dataset_name == "qm7b":
            inner_dataset = MyQM7bDataset()
        elif dataset_name.startswith("qm9"):
            label_keys = None
            cutoff = 5.0
            for param in params:
                name, value = param.split("=")
                if name == "label_keys":
                    label_keys = value.split("+")
                elif name == "cutoff":
                    cutoff = float(value)
            inner_dataset = MyQM9Dataset(label_keys=label_keys, cutoff=cutoff)
        elif dataset_name.startswith("qm9edge"):
            label_keys = None
            for param in params:
                name, value = param.split("=")
                if name == "label_keys":
                    label_keys = value.split("+")
            inner_dataset = MyQM9EdgeDataset(label_keys=label_keys)
        elif dataset_name.startswith("minigc"):
            num_graphs = None
            min_num_v = None
            max_num_v = None
            data_seed = seed
            for param in params:
                name, value = param.split("=")
                if name == "num_graphs":
                    num_graphs = int(value)
                elif name == "min_num_v":
                    min_num_v = int(value)
                elif name == "max_num_v":
                    max_num_v = int(value)
                elif name == "seed":
                    data_seed = int(value)
            inner_dataset = MyMiniGCDataset(
                num_graphs, min_num_v, max_num_v, seed=data_seed
            )
        elif dataset_name.startswith("tu"):
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyTUDataset(name=nm)
        elif dataset_name.startswith("gin"):
            nm = None
            self_loop = None
            degree_as_nlabel = False
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
                elif name == "self_loop":
                    if value.lower() == "false":
                        self_loop = False
                    elif value.lower() == "true":
                        self_loop = True
                elif name == "degree_as_nlabel":
                    if value.lower() == "false":
                        degree_as_nlabel = False
                    elif value.lower() == "true":
                        degree_as_nlabel = True
            inner_dataset = MyGINDataset(
                name=nm, self_loop=self_loop, degree_as_nlabel=degree_as_nlabel
            )
        elif dataset_name.startswith("fakenews"):
            nm = None
            feature_name = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
                elif name == "feature_name":
                    feature_name = value
            inner_dataset = MyFakeNewsDataset(name=nm, feature_name=feature_name)
        else:
            raise ValueError(f"Unknown dataset specificaion {dataset_name}")

        return (
            None
            if inner_dataset is None
            else GraphormerDGLDataset(inner_dataset, seed)
        )
