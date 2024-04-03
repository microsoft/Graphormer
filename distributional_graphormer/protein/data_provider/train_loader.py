import itertools
import os

import numpy as np
import torch
from common import config as cfg
from torch.utils.data import DataLoader

from .dataset import StructureDataset, StructureDatasetNPY
from .util import LMDBReader


class TrainSampler:
    def __init__(self):
        pass

    def __iter__(self):
        batch_size = cfg.max_tokens_per_gpu // cfg.max_tokens_per_sample
        it = self._infinite()
        while True:
            chunk = tuple(itertools.islice(it, batch_size))
            yield chunk

    def _infinite(self):
        train_list_path = os.path.join(cfg.dataset_dir, "list", "pdb_all.list")
        train_list = open(train_list_path).readlines()
        train_list = [_.strip() for _ in train_list]
        ids = np.arange(len(train_list))
        while True:
            c = np.random.choice(ids)
            yield train_list[c]


class TrainLoader(DataLoader):
    def __init__(self):
        self._dataset = StructureDatasetNPY()
        self._sampler = TrainSampler()
        super().__init__(
            self._dataset,
            batch_size=None,
            sampler=self._sampler,
            num_workers=4,
            pin_memory=True,
        )
