from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class RBF(nn.Module):
    def __init__(self, K=128, edge_types=1024):
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
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class Graph3DBias(nn.Module):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(
        self,
        num_heads,
        num_atom_types,
        num_layers,
        embed_dim,
        num_kernel,
        dist_feature_extractor,
        no_share_rpe=False,
    ):
        super(Graph3DBias, self).__init__()
        self.num_heads = num_heads
        self.num_atom_types = num_atom_types
        self.num_layers = num_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        num_edge_types = num_atom_types * num_atom_types

        rpe_heads = (
            self.num_heads * self.num_layers if self.no_share_rpe else self.num_heads
        )
        if dist_feature_extractor == "gbf" and dist_feature_extractor == "rbf":
            raise ValueError("dist_feature_extractor can only be gbf or rbf")
        self.dist_feature_extractor = GaussianLayer(self.num_kernel, num_edge_types) if dist_feature_extractor == "gbf" else RBF(self.num_kernel, num_edge_types)
        self.feature_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

    def forward(self, batched_data, crystal=False):
        pos, x = (
            batched_data["pos"] if not crystal else batched_data["crystal_pos"],
            batched_data["x"],
        )  # pos shape: [n_graphs, n_nodes, 3]
        n_graph, n_node, _ = pos.shape

        atoms = x[:, :, 0]
        edge_types = atoms.view(n_graph, n_node, 1) * self.num_atom_types + atoms.view(
            n_graph, 1, n_node
        )
        padding_mask = atoms.eq(0)  # (G, T)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2) # (G, T, T, 3)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node) # (G, T, T)
        # delta_pos /= dist.unsqueeze(-1) + 1e-5 # (G, T, T, 3)
        delta_pos_norm = delta_pos / (dist.unsqueeze(-1) + 1e-5)

        edge_feature = self.dist_feature_extractor(dist, edge_types) # (G, T, T, K)

        graph_attn_bias = self.feature_proj(edge_feature) # (G, T, T, H)

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous() # (G, H, T, T)
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        if crystal:
            return {
                "attn_bias_3d_crystal": graph_attn_bias,
                "edge_features_crystal": merge_edge_features,
            }
        else:
            return {
                "dist": dist,
                "delta_pos": delta_pos_norm,
                "attn_bias_3d": graph_attn_bias,
                "edge_features": merge_edge_features,
            }


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


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
        self.force_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1, bias=False)

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
        f1 = self.force_proj(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force
