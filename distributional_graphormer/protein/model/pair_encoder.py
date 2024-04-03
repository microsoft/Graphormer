import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .attention import FeedForward, MultiHeadAttention


class TriangleAttention(nn.Module):
    def __init__(self, d_model, d_key, n_head, dropout=0.1):
        super(TriangleAttention, self).__init__()
        self.projection = nn.Linear(d_model, n_head, bias=False)
        self.mha = MultiHeadAttention(
            d_model=d_model, d_key=d_key, n_head=n_head, dropout=dropout
        )

    def forward(self, x, bias):
        bias = bias + self.projection(x).permute(0, 3, 1, 2).unsqueeze(1)
        return self.mha(x, bias)[0]


class TriangleMultiplication(nn.Module):
    def __init__(self, d_model, d_key, dropout):
        super(TriangleMultiplication, self).__init__()
        self.l_proj = nn.Linear(d_model, d_key, bias=False)
        self.r_proj = nn.Linear(d_model, d_key, bias=False)
        self.l_gate = nn.Sequential(nn.Linear(d_model, d_key, bias=False), nn.Sigmoid())
        self.r_gate = nn.Sequential(nn.Linear(d_model, d_key, bias=False), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_key)
        self.o_proj = nn.Linear(d_key, d_model, bias=False)
        self.o_gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        mask = 1.0 - mask.to(x)
        mask = mask[:, None] * mask[:, :, None]
        mask = mask.unsqueeze(-1)
        l_x = self.l_proj(x)
        l_g = self.l_gate(x)
        l_x = mask * l_x * l_g

        r_x = self.r_proj(x)
        r_g = self.r_gate(x)
        r_x = mask * r_x * r_g

        o = torch.einsum("bikc,bjkc->bijc", l_x, r_x)
        o = self.o_proj(self.norm(o)) * self.o_gate(x)
        o = self.dropout(o)
        return o


class TriangleLayer(nn.Module):
    def __init__(self, d_model, d_key, n_head, dim_feedforward, dropout):
        super(TriangleLayer, self).__init__()
        self.row_norm = nn.LayerNorm(d_model)
        self.row_attn = TriangleAttention(d_model, d_key, n_head, dropout)
        self.col_norm = nn.LayerNorm(d_model)
        self.col_attn = TriangleAttention(d_model, d_key, n_head, dropout)
        self.i_norm = nn.LayerNorm(d_model)
        self.i_mul = TriangleMultiplication(d_model, d_key, dropout=0.3)
        self.o_norm = nn.LayerNorm(d_model)
        self.o_mul = TriangleMultiplication(d_model, d_key, dropout=0.3)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x, mask):
        bias = mask[:, None, None, None]
        bias = bias.float().masked_fill(bias, float("-inf"))
        x = x + self.row_attn(self.row_norm(x), bias)
        x = x.transpose(-2, -3)
        x = x + self.col_attn(self.col_norm(x), bias)
        x = x + self.i_mul(self.i_norm(x), mask)
        x = x.transpose(-2, -3)
        x = x + self.o_mul(self.o_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PairEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(PairEncoder, self).__init__()
        self.layers = nn.ModuleList([TriangleLayer(**kwargs) for _ in range(n_layer)])

    def forward(self, x, mask):
        for module in self.layers:
            if not self.training:
                x = module(x, mask)
            else:
                x = torch.utils.checkpoint.checkpoint(module, x, mask)
        return x
