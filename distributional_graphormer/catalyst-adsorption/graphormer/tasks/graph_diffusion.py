import logging
import math
import os

import torch
import contextlib
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, open_dict, OmegaConf

import numpy as np
from fairseq.data import (
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
)

from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

from ..data.diff_datasets.diff_dataset import (
    EpochShuffleDataset,
)

from ..data.diff_datasets.oc_dataset import (
    build_oc_kde_dm,
    build_oc_kde_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphDiffusionConfig(FairseqDataclass):
    """
    Data options
    """

    data_path: str = field(
        default="",
        metadata={"help": "path to dataset directory"},
    )

    data_seed: int = field(
        default=2022,
        metadata={"help": "random seed for data split"},
    )

    max_nodes: int = field(
        default=128,
        metadata={"help": "max nodes per graph"},
    )

    seed: int = II("common.seed")


@register_task("graph_diffusion", dataclass=GraphDiffusionConfig)
class GraphDiffusionTask(FairseqTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dm = build_oc_kde_dm(cfg.data_path)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        raw_dataset = self.dm.get_split(split)
        batched_data = build_oc_kde_dataset(raw_dataset)

        sizes_np = np.array([1] * len(batched_data))  # FIXME

        dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
            },
            sizes=sizes_np,
        )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        if split == "train":
            dataset = EpochShuffleDataset(dataset, self.cfg.seed)

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes

        model = models.build_model(cfg, self)

        return model

    def max_nodes(self):
        return self.cfg.max_nodes

    def max_positions(self):
        return self.cfg.max_nodes

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
