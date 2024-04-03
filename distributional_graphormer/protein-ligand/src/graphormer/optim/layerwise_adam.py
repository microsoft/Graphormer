import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import re
import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import register_optimizer
from fairseq.optim.adam import FairseqAdamConfig, FairseqAdam

from omegaconf import II, OmegaConf

logger = logging.getLogger(__name__)

@dataclass
class LayerwiseAdamConfig(FairseqDataclass):
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )

    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")

    lr_scale_decay: float = field(default=0.65)

@register_optimizer("layerwise_adam", dataclass=LayerwiseAdamConfig)
class LayerwiseAdam(FairseqAdam):
    def __init__(self, cfg: FairseqAdamConfig, params):
        self.cfg = cfg
        self._optimizer = Adam(params, lr_scale_decay=cfg.lr_scale_decay, **self.optimizer_config)


class LRManager:
    def __init__(self, params, lr_scale_decay: float):
        self.lr_scale_decay = lr_scale_decay

        layer_numbers = []
        for p in params:
            name = p.param_group
            layer_numbers.append(self.get_layer_num(name))

        self.layer_numbers = []
        max_layer_number = max(layer_numbers)
        for n in layer_numbers:
            if n == -1:
                n = max_layer_number + 1
            self.layer_numbers.append(n)

        self.lr_scales = []
        for n in self.layer_numbers:
            self.lr_scales.append(self.lr_scale_decay ** (max_layer_number + 1 - n))

    def get_layer_num(self, name):
        """
        prefix: encoder.graph_encoder.
        keys:
            [0] graph_node_feature: node embedding
            [0] graph_attn_bias: bias
            [0] emb_layer_norm: norm0
            [1-12] layers.[0-11]
            [13] head
        """
        # is_bias = name.endswith(".bias")
        # in_layer_norm = "layer_norm." in name
        in_embedding_layers = any(
            k in name
            for k in {"graph_node_feature.", "graph_attn_bias.", "emb_layer_norm."}
        )
        backbone_number = re.findall(r"layers\.(\d+)\.", name)
        in_backbone = len(backbone_number) > 0
        if in_backbone:
            backbone_number = int(backbone_number[0])

        # get lr decay
        """
        prefix: encoder.graph_encoder.
        keys:
            [0] graph_node_feature: node embedding
            [0] graph_attn_bias: bias
            [0] emb_layer_norm: norm0
            [1-12] layers.[0-11]
            [13] head
        """
        if in_embedding_layers:
            layer_number = 0
        elif in_backbone:
            layer_number = backbone_number + 1
        else:  # in downstream head
            layer_number = -1

        return layer_number


class Adam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr_scale_decay,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(Adam, self).__init__(params, defaults)
        for group in self.param_groups:
            lr_manager = LRManager(list(group["params"]), lr_scale_decay=lr_scale_decay)
            for idx, p in enumerate(group["params"]):
                p.lr_scale = lr_manager.lr_scales[idx]

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"]
                    * p.lr_scale  # layerwise lr scale
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss
