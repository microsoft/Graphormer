# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("binary_logloss", dataclass=FairseqDataclass)
class GraphPredictionBinaryLogLoss(FairseqCriterion):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])
        preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
        
        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[: logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)
        loss = functional.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": torch.sum(mask.type(torch.int64)),
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": (preds == targets[:preds.size(0)]).sum(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("binary_logloss_with_flag", dataclass=FairseqDataclass)
class GraphPredictionBinaryLogLossWithFlag(GraphPredictionBinaryLogLoss):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
        
        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[: logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)
        loss = functional.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": (preds == targets[:preds.size(0)]).sum(),
        }
        return loss, sample_size, logging_output
