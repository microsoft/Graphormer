"""A self-contained Graphormer-style model for benchmarking.

The reference microsoft/Graphormer model ships inside fairseq and
relies on a long stack of CUDA extensions. Importing it for the
sole purpose of a robustness benchmark is heavy and brittle. This
module implements a *minimal* Graphormer-shaped classifier in plain
PyTorch so that the benchmark runs in any environment with ``torch``
and ``numpy``.

The architecture mirrors the Graphormer paper in spirit:

* Each node starts with a learned embedding of its in-degree.
* A stack of Graphormer layers applies multi-head self-attention
  biased by the spatial-positional distance matrix.
* A virtual ``[CLS]``-style token is prepended; its final hidden
  state is the graph representation.
* A two-layer MLP head produces class logits.

The bias-tensor convention matches the adapter: ``attn_bias`` has
shape ``(B, N+1, N+1)`` and indexes a learned distance embedding.
Indices equal to ``-1`` are masked out; index ``0`` is the virtual
node's slot.

This module is intentionally simple. The point of the benchmark is
not to win leaderboard scores; it is to put a *graph transformer*
under the same robustness microscope that we put the original GNN
under, and to show that the topological blind spot is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GraphormerConfig:
    """Configuration for :class:`GraphormerClassifier`."""

    n_classes: int = 2
    hidden_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    max_distance: int = 32
    dropout: float = 0.1


class GraphormerLayer(nn.Module):
    """One pre-norm Graphormer block with spatial-bias attention."""

    def __init__(self, hidden_dim: int, n_heads: int, max_distance: int, dropout: float):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.spatial_emb = nn.Embedding(max_distance + 2, n_heads)

    def forward(
        self,
        x: Tensor,
        attn_bias_idx: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        h = self.norm1(x)
        b, n, d = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        bias = self.spatial_emb(attn_bias_idx.clamp(min=0))
        bias = bias.permute(0, 3, 1, 2)
        attn_logits = attn_logits + bias
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        out = self.out(out)
        x = x + self.resid_drop(out)

        h = self.norm2(x)
        x = x + self.resid_drop(self.ffn(h))
        return x


class GraphormerClassifier(nn.Module):
    """Minimal Graphormer-style graph classifier.

    The forward signature mirrors the keys produced by
    :func:`graphormer_redteam.adapter.collate` so that the model can
    be wired into the training loop without any glue code.
    """

    def __init__(self, config: GraphormerConfig | None = None):
        super().__init__()
        self.config = config or GraphormerConfig()
        c = self.config

        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.in_degree_emb = nn.Embedding(256, c.hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphormerLayer(c.hidden_dim, c.n_heads, c.max_distance, c.dropout)
                for _ in range(c.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(c.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.n_classes),
        )

    def forward(self, batch: dict) -> Tensor:
        x_idx = batch["in_deg"].clamp(min=0, max=255).long()
        h = self.in_degree_emb(x_idx)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)

        attn_bias = batch["attn_bias"]
        attn_mask = attn_bias < 0
        bias_idx = attn_bias.clamp(min=0)
        for layer in self.layers:
            h = layer(h, bias_idx, attn_mask=attn_mask)
        h = self.norm(h)
        return self.head(h[:, 0])
