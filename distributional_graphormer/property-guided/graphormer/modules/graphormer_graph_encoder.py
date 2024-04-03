# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from graphormer.modules.droppath import DropPath

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias
from .graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerGraphEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        # for Graphormer
        num_atoms = args.num_atoms
        num_in_degree = args.num_in_degree
        num_out_degree = args.num_out_degree
        num_edges = args.num_edges
        num_spatial = args.num_spatial
        num_edge_dis = args.num_edge_dis
        edge_type = args.edge_type
        multi_hop_max_dist = args.multi_hop_max_dist
        sandwich_norm = args.sandwich_norm
        num_encoder_layers = args.encoder_layers
        embedding_dim = args.encoder_embed_dim
        ffn_embedding_dim = args.encoder_ffn_embed_dim
        num_attention_heads = args.encoder_attention_heads

        # Fine-tuning parameters
        layer_scale = args.layer_scale
        droppath_prob = args.droppath_prob

        # Attention parameters
        dropout = args.dropout
        attention_dropout = args.attention_dropout
        activation_dropout = args.act_dropout
        encoder_normalize_before = args.encoder_normalize_before
        apply_graphormer_init = args.apply_graphormer_init
        activation_fn = args.activation_fn
        use_bonds = args.use_bonds

        # Disable original Dropout when using DropPath
        if droppath_prob > 0.0:
            dropout = attention_dropout = activation_dropout = 0.0

        # stochastic depth decay rule (linearly increasing)
        droppath_probs = [
            x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
        ]

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            use_bonds=use_bonds,
        )
        self.init_extra_node_layers(args)

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )
        self.init_extra_bias_layers(args)

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    layer_scale=layer_scale,
                    droppath=droppath_probs[i],
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    sandwich_norm=sandwich_norm,
                )
                for i in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        layer_scale,
        droppath,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        sandwich_norm,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            layer_scale=layer_scale,
            droppath=droppath,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            sandwich_norm=sandwich_norm,
        )

    def init_extra_node_layers(self, args):
        pass

    def init_extra_bias_layers(self, args):
        pass

    def forward_extra_node_layers(self, batched_data, x):
        """
        input:
            batched_data: dict
            x: tensor, B x T x C (T = N + 1)
        output:
            x: tensor, B x T x C (T = N + 1)
        """
        return x

    def forward_extra_bias_layers(self, batched_data, attn_bias):
        """
        attn_bias: B x H x T x T (T = N + 1)
        input:
            batched_data: dict
            attn_bias: tensor, B x H x T x T (T = N + 1)
        output:
            attn_bias: tensor, B x H x T x T (T = N + 1)
        """
        return attn_bias

    def make_padding_mask(self, batched_data):
        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x N x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        return padding_mask.contiguous()

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # B x T x 1, T = N + 1
        padding_mask = self.make_padding_mask(batched_data)

        # node features -> B x T x C
        x = self.graph_node_feature(batched_data)
        # extra layers -> keep the same shape
        x = self.forward_extra_node_layers(batched_data, x)

        # attn bias -> B x H x T x T
        attn_bias = self.graph_attn_bias(batched_data)
        # extra layers -> keep the same shape
        attn_bias = self.forward_extra_bias_layers(batched_data, attn_bias)

        if perturb is not None:
            # perturb: B x N x C
            x[:, 1:, :] += perturb

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        inner_attns = []

        if not last_state_only:
            inner_states.append(x)

        expand_mask = batched_data.get("expand_mask", None)
        outcell_index = batched_data.get("outcell_index", None)

        for layer in self.layers:
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
                need_weights=need_attn,
                expand_mask=expand_mask,
                outcell_index=outcell_index,
            )
            if not last_state_only:
                inner_states.append(x)
                if need_attn:
                    inner_attns.append(attn)

        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]
            if need_attn:
                inner_attns = [attn]

        if not need_attn:
            return inner_states, graph_rep
        else:
            return inner_states, graph_rep, inner_attns
