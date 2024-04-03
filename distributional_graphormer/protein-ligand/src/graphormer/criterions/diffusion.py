# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from dataclasses import dataclass, field

from fairseq.dataclass.configs import FairseqDataclass

import logging
import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from tqdm import tqdm
import numpy as np
import pickle as pkl
import os
import time

@dataclass
class DiffusionLossConfig(FairseqDataclass):
    valid_times: int = field(
        default=1, metadata={"help": "number of times to run validation"}
    )


@register_criterion("difussion_loss", dataclass=DiffusionLossConfig)
class DiffusionLoss(FairseqCriterion):
    def __init__(self, task, valid_times=1):
        super().__init__(task)
        self.valid_times = valid_times

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]

        if model.training:
            # output = model.get_training_output(**sample["net_input"])
            x = torch.randn(3, 3, requires_grad=True).mean()
            return x, 1, {'loss': torch.tensor(0), 'sample_size': 1, 'nsentences': 1, 'rmsd': torch.tensor(0), 'rmsd_lig': torch.tensor(0), 'rmsd_pro': torch.tensor(0)}
        else:
            with torch.no_grad():
                output = model.get_sampling_output(
                    **sample["net_input"], sampling_times=self.valid_times
                )

        persample_loss = output["persample_loss"]
        loss = torch.sum(persample_loss)
        sample_size = output["sample_size"]

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
        }

        if "persample_rmsd" in output:
            persample_rmsd = output["persample_rmsd"]
            rmsd = torch.sum(persample_rmsd)
            logging_output["rmsd"] = rmsd.data

        if "persample_rmsd_lig" in output and output['persample_rmsd_lig'] is not None:
            persample_rmsd_lig = output['persample_rmsd_lig']
            rmsd_lig = torch.sum(persample_rmsd_lig)
            logging_output['rmsd_lig'] = rmsd_lig.data
        if "persample_rmsd_pro" in output and output['persample_rmsd_pro'] is not None:
            persample_rmsd_pro = output['persample_rmsd_pro']
            rmsd_pro = torch.sum(persample_rmsd_pro)
            logging_output['rmsd_pro'] = rmsd_pro.data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        rmsd_sum = sum(log.get("rmsd", 0) for log in logging_outputs)
        rmsd_lig_sum = sum(log.get("rmsd_lig", 0) for log in logging_outputs)
        rmsd_pro_sum = sum(log.get("rmsd_pro", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        if rmsd_sum > 0:
            metrics.log_scalar("rmsd", rmsd_sum / sample_size, sample_size, round=6)
        if rmsd_lig_sum > 0:
            metrics.log_scalar("rmsd_lig", rmsd_lig_sum / sample_size, sample_size, round=6)
        if rmsd_pro_sum > 0:
            metrics.log_scalar("rmsd_pro", rmsd_pro_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

