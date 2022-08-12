# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import contextlib
from dataclasses import dataclass, field
from omegaconf import II, open_dict, OmegaConf
import importlib

import numpy as np
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

from graphormer.pretrain import load_pretrained_model

from ..data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    GraphormerDataset,
    EpochShuffleDataset,
)

import torch
from fairseq.optim.amp_optimizer import AMPOptimizer
import math

from ..data import DATASET_REGISTRY
import sys
import os

logger = logging.getLogger(__name__)


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    dataset_name: str = field(
        default="pcqm4m",
        metadata={"help": "name of the dataset"},
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )

    max_nodes: int = field(
        default=128,
        metadata={"help": "max nodes per graph"},
    )

    dataset_source: str = field(
        default="pyg",
        metadata={"help": "source of graph dataset, can be: pyg, dgl, ogb, smiles"},
    )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    seed: int = II("common.seed")

    pretrained_model_name: str = field(
        default="none",
        metadata={"help": "name of used pretrained model"},
    )

    load_pretrained_model_output_layer: bool = field(
        default=False,
        metadata={"help": "whether to load the output layer of pretrained model"},
    )

    train_epoch_shuffle: bool = field(
        default=False,
        metadata={"help": "whether to shuffle the dataset at each epoch"},
    )

    user_data_dir: str = field(
        default="",
        metadata={"help": "path to the module of user-defined dataset"},
    )


@register_task("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionTask(FairseqTask):
    """
    Graph prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.user_data_dir != "":
            self.__import_user_defined_datasets(cfg.user_data_dir)
            if cfg.dataset_name in DATASET_REGISTRY:
                dataset_dict = DATASET_REGISTRY[cfg.dataset_name]
                self.dm = GraphormerDataset(
                    dataset=dataset_dict["dataset"],
                    dataset_source=dataset_dict["source"],
                    train_idx=dataset_dict["train_idx"],
                    valid_idx=dataset_dict["valid_idx"],
                    test_idx=dataset_dict["test_idx"],
                    seed=cfg.seed)
            else:
                raise ValueError(f"dataset {cfg.dataset_name} is not found in customized dataset module {cfg.user_data_dir}")
        else:
            self.dm = GraphormerDataset(
                dataset_spec=cfg.dataset_name,
                dataset_source=cfg.dataset_source,
                seed=cfg.seed,
            )

    def __import_user_defined_datasets(self, dataset_dir):
        dataset_dir = dataset_dir.strip("/")
        module_parent, module_name = os.path.split(dataset_dir)
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
        for file in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, file)
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
            ):
                task_name = file[: file.find(".py")] if file.endswith(".py") else file
                importlib.import_module(module_name + "." + task_name)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(
            batched_data,
            max_node=self.max_nodes(),
            multi_hop_max_dist=self.cfg.multi_hop_max_dist,
            spatial_pos_max=self.cfg.spatial_pos_max,
        )

        data_sizes = np.array([self.max_nodes()] * len(batched_data))

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": target,
            },
            sizes=data_sizes,
        )

        if split == "train" and self.cfg.train_epoch_shuffle:
            dataset = EpochShuffleDataset(
                dataset, num_samples=len(dataset), seed=self.cfg.seed
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

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

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None


@dataclass
class GraphPredictionWithFlagConfig(GraphPredictionConfig):
    flag_m: int = field(
        default=3,
        metadata={
            "help": "number of iterations to optimize the perturbations with flag objectives"
        },
    )

    flag_step_size: float = field(
        default=1e-3,
        metadata={
            "help": "learing rate of iterations to optimize the perturbations with flag objective"
        },
    )

    flag_mag: float = field(
        default=1e-3,
        metadata={"help": "magnitude bound for perturbations in flag objectives"},
    )


@register_task("graph_prediction_with_flag", dataclass=GraphPredictionWithFlagConfig)
class GraphPredictionWithFlagTask(GraphPredictionTask):
    """
    Graph prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.flag_m = cfg.flag_m
        self.flag_step_size = cfg.flag_step_size
        self.flag_mag = cfg.flag_mag

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        batched_data = sample["net_input"]["batched_data"]["x"]
        n_graph, n_node = batched_data.shape[:2]
        perturb_shape = n_graph, n_node, model.encoder_embed_dim
        if self.flag_mag > 0:
            perturb = (
                torch.FloatTensor(*perturb_shape)
                .uniform_(-1, 1)
                .to(batched_data.device)
            )
            perturb = perturb * self.flag_mag / math.sqrt(perturb_shape[-1])
        else:
            perturb = (
                torch.FloatTensor(*perturb_shape)
                .uniform_(-self.flag_step_size, self.flag_step_size)
                .to(batched_data.device)
            )
        perturb.requires_grad_()
        sample["perturb"] = perturb
        with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
            loss, sample_size, logging_output = criterion(
                model, sample
            )
            if ignore_grad:
                loss *= 0
        loss /= self.flag_m
        total_loss = 0
        for _ in range(self.flag_m - 1):
            optimizer.backward(loss)
            total_loss += loss.detach()
            perturb_data = perturb.detach() + self.flag_step_size * torch.sign(
                perturb.grad.detach()
            )
            if self.flag_mag > 0:
                perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
                exceed_mask = (perturb_data_norm > self.flag_mag).to(perturb_data)
                reweights = (
                    self.flag_mag / perturb_data_norm * exceed_mask
                    + (1 - exceed_mask)
                ).unsqueeze(-1)
                perturb_data = (perturb_data * reweights).detach()
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            sample["perturb"] = perturb
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(
                    model, sample
                )
                if ignore_grad:
                    loss *= 0
            loss /= self.flag_m
        optimizer.backward(loss)
        total_loss += loss.detach()
        logging_output["loss"] = total_loss
        return total_loss, sample_size, logging_output
