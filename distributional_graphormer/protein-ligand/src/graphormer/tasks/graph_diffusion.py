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

from ..data.diff_datasets.md_dataset import (
    build_md_kde_dm,
    build_md_kde_dataset,
    EpochShuffleDataset,
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

    uses_ema: bool = field(
        default=False,
        metadata={"help": "whether to use EMA for model parameters"},
    )

    seed: int = II("common.seed")

    reweighting_file: str = field(
        default="",
        metadata={"help": "using reweighting file to reweight the loss according to RMSD_to_crystal"},
    )

    train_subset: str = field(
        default="train",
        metadata={"help": "which subset to use for training"},
    )
    valid_subset: str = field(
        default="valid",
        metadata={"help": "which subset to use for validation"},
    )
    need_all_poses: bool = field(
        default=False,
        metadata={"help": "for kde and flowode, they need to use all poses for a system"},
    )



@register_task("graph_diffusion", dataclass=GraphDiffusionConfig)
class GraphDiffusionTask(FairseqTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dm = build_md_kde_dm(cfg.data_path, cfg.reweighting_file, cfg.need_all_poses)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        raw_dataset = self.dm.get_split(split, self.cfg.train_subset)
        batched_data = build_md_kde_dataset(raw_dataset)

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

        if  split in self.cfg.train_subset:
            dataset = EpochShuffleDataset(dataset, self.cfg.seed)

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes

        model = models.build_model(cfg, self)

        return model

    def train_step(
        self,
        sample,
        model,
        criterion,
        optimizer,
        update_num,
        ignore_grad=False,
        ema_model=None,
    ):
        return super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

    def valid_step(self, sample, model, criterion, ema_model=None):
        if self.cfg.uses_ema:
            assert ema_model is not None
            model = ema_model

        return super().valid_step(sample, model, criterion)

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

    @property
    def uses_ema(self):
        return self.cfg.uses_ema
