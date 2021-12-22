# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Mapping, Sequence, Tuple
from numpy import mod
import torch
from torch import Tensor
import torch.nn.functional as F


from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("mae_deltapos")
class IS2RECriterion(FairseqCriterion):
    e_thresh = 0.02
    e_mean = -1.4729953244844094
    e_std = 2.2707848125378405
    d_mean = [0.1353900283575058, 0.06877671927213669, 0.08111362904310226]
    d_std = [1.7862379550933838, 1.78688645362854, 0.8023099899291992]

    def __init__(self, task, cfg):
        super().__init__(task)
        self.node_loss_weight = cfg.node_loss_weight
        self.min_node_loss_weight = cfg.min_node_loss_weight
        self.max_update = cfg.max_update
        self.node_loss_weight_range = max(
            0, self.node_loss_weight - self.min_node_loss_weight
        )

    def forward(
        self,
        model: Callable[..., Tuple[Tensor, Tensor, Tensor]],
        sample: Mapping[str, Mapping[str, Tensor]],
        reduce=True,
    ):
        update_num = model.num_updates
        assert update_num >= 0
        node_loss_weight = (
            self.node_loss_weight
            - self.node_loss_weight_range * update_num / self.max_update
        )

        valid_nodes = sample["net_input"]["atoms"].ne(0).sum()
        output, node_output, node_target_mask = model(
            **sample["net_input"],
        )

        relaxed_energy = sample["targets"]["relaxed_energy"]
        relaxed_energy = relaxed_energy.float()
        relaxed_energy = (relaxed_energy - self.e_mean) / self.e_std
        sample_size = relaxed_energy.numel()
        loss = F.l1_loss(output.float().view(-1), relaxed_energy, reduction="none")
        with torch.no_grad():
            energy_within_threshold = (loss.detach() * self.e_std < self.e_thresh).sum()
        loss = loss.sum()

        deltapos = sample["targets"]["deltapos"].float()
        deltapos = (deltapos - deltapos.new_tensor(self.d_mean)) / deltapos.new_tensor(
            self.d_std
        )
        deltapos *= node_target_mask
        node_output *= node_target_mask
        target_cnt = node_target_mask.sum(dim=[1, 2])
        node_loss = (
            F.l1_loss(node_output.float(), deltapos, reduction="none")
            .mean(dim=-1)
            .sum(dim=-1)
            / target_cnt
        ).sum()

        logging_output = {
            "loss": loss.detach(),
            "energy_within_threshold": energy_within_threshold,
            "node_loss": node_loss.detach(),
            "sample_size": sample_size,
            "nsentences": sample_size,
            "num_nodes": valid_nodes.detach(),
            "node_loss_weight": node_loss_weight * sample_size,
        }
        return loss + node_loss_weight * node_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs: Sequence[Mapping]) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        energy_within_threshold_sum = sum(
            log.get("energy_within_threshold", 0) for log in logging_outputs
        )
        node_loss_sum = sum(log.get("node_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        mean_loss = (loss_sum / sample_size) * IS2RECriterion.e_std
        energy_within_threshold = energy_within_threshold_sum / sample_size
        mean_node_loss = (node_loss_sum / sample_size) * sum(IS2RECriterion.d_std) / 3.0
        mean_n_nodes = (
            sum([log.get("num_nodes", 0) for log in logging_outputs]) / sample_size
        )
        node_loss_weight = (
            sum([log.get("node_loss_weight", 0) for log in logging_outputs])
            / sample_size
        )

        metrics.log_scalar("loss", mean_loss, sample_size, round=6)
        metrics.log_scalar("ewth", energy_within_threshold, sample_size, round=6)
        metrics.log_scalar("node_loss", mean_node_loss, sample_size, round=6)
        metrics.log_scalar("nodes_per_graph", mean_n_nodes, sample_size, round=6)
        metrics.log_scalar("node_loss_weight", node_loss_weight, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
