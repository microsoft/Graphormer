from functools import lru_cache

import ogb
import numpy as np
import torch
from torch.nn import functional as F
from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset

from .wrapper import MyPygGraphPropPredDataset
from .collator import collator

from typing import Optional, Union
from torch_geometric.data import Data as PYGDataset
from dgl.data import DGLDataset
from .dgl_datasets import DGLDatasetLookupTable, GraphormerDGLDataset
from .pyg_datasets import PYGDatasetLookupTable, GraphormerPYGDataset
from .ogb_datasets import OGBDatasetLookupTable


class BatchedDataDataset(FairseqDataset):
    def __init__(
        self, dataset, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collator(
            samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)


class GraphormerDataset:
    def __init__(
        self,
        dataset: Optional[Union[PYGDataset, DGLDataset]] = None,
        dataset_spec: Optional[str] = None,
        dataset_source: Optional[str] = None,
        seed: int = 0,
        train_idx = None,
        valid_idx = None,
        test_idx = None,
    ):
        super().__init__()
        if dataset is not None:
            if dataset_source == "dgl":
                self.dataset = GraphormerDGLDataset(dataset, seed=seed, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
            elif dataset_source == "pyg":
                self.dataset = GraphormerPYGDataset(dataset, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
            else:
                raise ValueError("customized dataset can only have source pyg or dgl")
        elif dataset_source == "dgl":
            self.dataset = DGLDatasetLookupTable.GetDGLDataset(dataset_spec, seed=seed)
        elif dataset_source == "pyg":
            self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_spec, seed=seed)
        elif dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_spec, seed=seed)
        self.setup()

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.num_samples)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
