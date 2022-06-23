# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
A demonstrative file to show users to define and use their customized graphormer model.
You can define your own way to calculate the attention bias, or define your own prediction head,
by overriding specific functions.

Relation between classes:
- GraphormerCustomModel
  - GraphormerEncoder
    - GraphormerGraphEncoder
"""

import logging
import contextlib

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.utils import safe_hasattr
from ..modules import init_graphormer_params
from ..modules import PredictLayerGroup
from ..modules import GraphormerGraphEncoder as GraphormerGraphEncoderBase
from .graphormer_encoder import GraphormerEncoder as GraphormerEncoderBase

from ..utils import (
    graphormer_default_add_args,
    guess_fisrt_load,
    upgrade_state_dict_named_from_pretrained,
)


class GraphormerGraphEncoder(GraphormerGraphEncoderBase):
    """
    Define extra node layers or bias layers here if needed.
    """

    def init_extra_node_layers(self, args):
        super().init_extra_node_layers(args)
        # Your code here
        pass

    def init_extra_bias_layers(self, args):
        super().init_extra_bias_layers(args)
        # Your code here
        pass

    def forward_extra_node_layers(self, batched_data, x):
        x = super().forward_extra_node_layers(batched_data, x)
        # Your code here
        return x

    def forward_extra_bias_layers(self, batched_data, attn_bias):
        bias = super().forward_extra_bias_layers(batched_data, attn_bias)
        # Your code here
        return bias


class GraphormerEncoder(GraphormerEncoderBase):
    def build_graph_encoder(self, args):
        return GraphormerGraphEncoder(args)


@register_model("graphormer_custom")
class GraphormerCustomModel(FairseqEncoderModel):
    """
    Register your customized model architecture here.
    """
    
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)

        print(f"{self.__class__.__name__}: {self}")

    @staticmethod
    def add_args(parser):
        graphormer_default_add_args(parser)

    def max_nodes(self):
        return self.encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        encoder = GraphormerEncoder(args)
        return cls(args, encoder)

    # Define customized prediction layers.
    def register_predictor(self, out_dims):
        self.predictor = PredictLayerGroup(
            in_dim=self.args.encoder_embed_dim,
            out_dims=out_dims,
            activation=utils.get_activation_fn(self.args.activation_fn),
            n_layers=2,
        )

    def forward(self, batched_data, **kwargs):
        encoder_out = self.encoder(batched_data, **kwargs)
        x_cls = encoder_out["encoder_out"][0][0, :, :]  # B x d
        x = self.predictor(x_cls)
        return x

    def upgrade_state_dict_named(self, state_dict, name):
        named_parameters = {k: v for k, v in self.named_parameters()}
        first_load = guess_fisrt_load(named_parameters, state_dict, name)

        if first_load:
            msg = upgrade_state_dict_named_from_pretrained(
                named_parameters, state_dict, name
            )
            logger.warning(f"upgrade_state_dict_named_from_pretrained: {msg}")

            # fill missing keys
            for k in named_parameters:
                if k not in state_dict:
                    state_dict[k] = named_parameters[k]
                    logger.warning(
                        f"Warning: {k} is missing from the checkpoint, copying from model"
                    )

            # remove ununsed keys
            for k in list(state_dict.keys()):
                if k not in named_parameters:
                    del state_dict[k]
                    logger.warning(
                        f"Warning: {k} is not used in the model, removing from the checkpoint"
                    )

        return state_dict


@register_model_architecture("graphormer_custom", "graphormer_custom_base")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
