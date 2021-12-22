# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor = None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs = softmax_dropout(
            attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training
        ).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force


@register_model("graphormer3d")
class Graphormer3D(BaseFairseqModel):
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument("--blocks", type=int, metavar="L", help="num blocks")
        parser.add_argument(
            "--embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--min-node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--num-kernel",
            type=int,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.atom_types = 64
        self.edge_types = 64 * 64
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.args.embed_dim, padding_idx=0
        )
        self.tag_encoder = nn.Embedding(3, self.args.embed_dim)
        self.input_dropout = self.args.input_dropout
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    num_attention_heads=self.args.attention_heads,
                    dropout=self.args.dropout,
                    attention_dropout=self.args.attention_dropout,
                    activation_dropout=self.args.activation_dropout,
                )
                for _ in range(self.args.layers)
            ]
        )

        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(self.args.embed_dim)

        self.engergy_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.args.embed_dim, 1
        )
        self.energe_agg_factor: Callable[[Tensor], Tensor] = nn.Embedding(3, 1)
        nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)

        K = self.args.num_kernel

        self.rbf: Callable[[Tensor, Tensor], Tensor] = RBF(K, self.edge_types)
        self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
            K, self.args.attention_heads
        )
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(K, self.args.embed_dim)
        self.node_proc: Callable[[Tensor, Tensor, Tensor], Tensor] = NodeTaskHead(
            self.args.embed_dim, self.args.attention_heads
        )

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        return super().set_num_updates(num_updates)

    def forward(self, atoms: Tensor, tags: Tensor, pos: Tensor, real_mask: Tensor):
        padding_mask = atoms.eq(0)

        n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: Tensor = delta_pos.norm(dim=-1)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(
            n_graph, 1, n_node
        )

        rbf_feature = self.rbf(dist, edge_type)
        edge_features = rbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.tag_encoder(tags)
            + self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        output = output.transpose(0, 1).contiguous()

        graph_attn_bias = self.bias_proj(rbf_feature).permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.args.blocks):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        output = output.transpose(0, 1)

        eng_output = F.dropout(output, p=0.1, training=self.training)
        eng_output = (
            self.engergy_proj(eng_output) * self.energe_agg_factor(tags)
        ).flatten(-2)
        output_mask = (
            tags > 0
        ) & real_mask  # no need to consider padding, since padding has tag 0, real_mask False

        eng_output *= output_mask
        eng_output = eng_output.sum(dim=-1)

        node_output = self.node_proc(output, graph_attn_bias, delta_pos)

        node_target_mask = output_mask.unsqueeze(-1)
        return eng_output, node_output, node_target_mask


@register_model_architecture("graphormer3d", "graphormer3d_base")
def base_architecture(args):
    args.blocks = getattr(args, "blocks", 4)
    args.layers = getattr(args, "layers", 12)
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 768)
    args.attention_heads = getattr(args, "attention_heads", 48)
    args.input_dropout = getattr(args, "input_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.node_loss_weight = getattr(args, "node_loss_weight", 15)
    args.min_node_loss_weight = getattr(args, "min_node_loss_weight", 1)
    args.eng_loss_weight = getattr(args, "eng_loss_weight", 1)
    args.num_kernel = getattr(args, "num_kernel", 128)
