# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Dataset
from ..pyg_datasets import GraphormerPYGDataset
import torch.distributed as dist
import os

class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4MDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4MDataset, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyPygGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).process()
        if dist.is_initialized():
            dist.barrier()


class OGBDatasetLookupTable:
    @staticmethod
    def GetOGBDataset(dataset_name: str, seed: int) -> Optional[Dataset]:
        inner_dataset = None
        train_idx = None
        valid_idx = None
        test_idx = None
        if dataset_name == "ogbg-molhiv":
            folder_name = dataset_name.replace("-", "_")
            os.system(f"mkdir -p dataset/{folder_name}/")
            os.system(f"touch dataset/{folder_name}/RELEASE_v1.txt")
            inner_dataset = MyPygGraphPropPredDataset(dataset_name)
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
        elif dataset_name == "ogbg-molpcba":
            folder_name = dataset_name.replace("-", "_")
            os.system(f"mkdir -p dataset/{folder_name}/")
            os.system(f"touch dataset/{folder_name}/RELEASE_v1.txt")
            inner_dataset = MyPygGraphPropPredDataset(dataset_name)
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
        elif dataset_name == "pcqm4mv2":
            os.system("mkdir -p dataset/pcqm4m-v2/")
            os.system("touch dataset/pcqm4m-v2/RELEASE_v1.txt")
            inner_dataset = MyPygPCQM4Mv2Dataset()
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test-dev"]
        elif dataset_name == "pcqm4m":
            os.system("mkdir -p dataset/pcqm4m_kddcup2021/")
            os.system("touch dataset/pcqm4m_kddcup2021/RELEASE_v1.txt")
            inner_dataset = MyPygPCQM4MDataset()
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
        else:
            raise ValueError(f"Unknown dataset name {dataset_name} for ogb source.")
        return (
            None
            if inner_dataset is None
            else GraphormerPYGDataset(
                inner_dataset, seed, train_idx, valid_idx, test_idx
            )
        )
