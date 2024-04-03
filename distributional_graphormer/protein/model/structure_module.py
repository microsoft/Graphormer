import math

import torch
import torch.nn.functional as F
from common import config as cfg
from torch import nn

from .attention import FeedForward


class PoseHead(nn.Module):
    def __init__(self, ninp):
        super(PoseHead, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(ninp),
            nn.Linear(ninp, ninp),
            nn.ReLU(),
            nn.Linear(ninp, 9 + 50),
        )

    def forward(self, x, eps=1e-8):
        x = self.fc(x)
        plddt = x[..., 9:]
        x = x[..., :9].reshape(*x.shape[:-1], 3, 3)
        T = x[..., 0]
        v1, v2 = x[..., 1], x[..., 2]
        v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
        v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
        e1 = v1
        e2 = torch.cross(e1, v2)
        e3 = torch.cross(e1, e2)
        IR = torch.cat([e1, e2, e3], dim=-1).reshape(*e3.shape[:-1], 3, 3)
        return (T, IR), plddt


class DiffHead(nn.Module):
    def __init__(self, ninp):
        super(DiffHead, self).__init__()
        self.fc_t = nn.Sequential(
            nn.LayerNorm(ninp),
            nn.Linear(ninp, ninp),
            nn.ReLU(),
            nn.Linear(ninp, 3),
        )
        self.fc_eps = nn.Sequential(
            nn.LayerNorm(ninp),
            nn.Linear(ninp, ninp),
            nn.ReLU(),
            nn.Linear(ninp, 3),
        )

    def forward(self, x):
        T_eps = self.fc_t(x)
        IR_eps = self.fc_eps(x)
        return (T_eps, IR_eps)


class SAAttention(nn.Module):
    def __init__(self, d_model, d_pair, n_head, dropout=0.1):
        super(SAAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError(
                "The hidden size is not a multiple of the number of attention heads"
            )
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.scalar_query = nn.Linear(d_model, d_model, bias=False)
        self.scalar_key = nn.Linear(d_model, d_model, bias=False)
        self.scalar_value = nn.Linear(d_model, d_model, bias=False)
        self.pair_bias = nn.Linear(d_pair, n_head, bias=False)
        self.point_query = nn.Linear(d_model, n_head * 3 * 4, bias=False)
        self.point_key = nn.Linear(d_model, n_head * 3 * 4, bias=False)
        self.point_value = nn.Linear(d_model, n_head * 3 * 8, bias=False)

        self.scalar_weight = 1.0 / math.sqrt(3 * self.d_k)
        self.point_weight = 1.0 / math.sqrt(3 * 4 * 9 / 2)
        self.trained_point_weight = nn.Parameter(torch.rand(n_head))
        self.pair_weight = 1.0 / math.sqrt(3)

        self.pair_value = nn.Linear(d_pair, d_model, bias=False)

        self.fc_out = nn.Linear(d_model * 2 + n_head * 8 * 4, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1d, x2d, pose, bias):
        T, R = pose[0], pose[1].transpose(-1, -2)

        # (B, L, nhead, C)
        q_scalar = self.scalar_query(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)
        k_scalar = self.scalar_key(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)
        v_scalar = self.scalar_value(x1d).reshape(*x1d.shape[:-1], self.n_head, -1)

        # (B, nhead, L, L)
        scalar_attn = torch.einsum(
            "bihc,bjhc->bhij", q_scalar * self.scalar_weight, k_scalar
        )

        # (B, L, nhead, num, 3)
        q_point_local = self.point_query(x1d).reshape(
            *x1d.shape[:-1], self.n_head, -1, 3
        )
        k_point_local = self.point_key(x1d).reshape(*x1d.shape[:-1], self.n_head, -1, 3)
        v_point_local = self.point_value(x1d).reshape(
            *x1d.shape[:-1], self.n_head, -1, 3
        )

        def apply_affine(point, T, R):
            return (
                torch.matmul(R[:, :, None, None], point.unsqueeze(-1)).squeeze(-1)
                + T[:, :, None, None]
            )

        # (B, L, nhead, num, 3)
        q_point_global = apply_affine(q_point_local, T, R)
        k_point_global = apply_affine(k_point_local, T, R)
        v_point_global = apply_affine(v_point_local, T, R)

        # (B, L, L, nhead, num)
        point_attn = torch.norm(
            q_point_global.unsqueeze(2) - k_point_global.unsqueeze(1), dim=-1
        )
        point_weight = self.point_weight * F.softplus(self.trained_point_weight)
        point_attn = (
            -0.5
            * point_weight[:, None, None]
            * torch.sum(point_attn, dim=-1).permute(0, 3, 1, 2)
        )

        pair_attn = self.pair_weight * self.pair_bias(x2d).permute(0, 3, 1, 2)

        attn_logits = scalar_attn + point_attn + pair_attn + bias

        # (B, nhead, L, L)
        attn = torch.softmax(attn_logits, dim=-1)

        out_scalar = torch.einsum("bhij,bjhc->bihc", attn, v_scalar)
        out_scalar = out_scalar.reshape(*out_scalar.shape[:2], -1)
        with torch.cuda.amp.autocast(enabled=False):
            out_point_global = torch.einsum(
                "bhij,bjhcp->bihcp", attn.float(), v_point_global.float()
            )
        out_point_local = torch.matmul(
            R.transpose(-1, -2)[:, :, None, None],
            (out_point_global - T[:, :, None, None]).unsqueeze(-1),
        ).squeeze(-1)

        out_point_norm = torch.norm(out_point_local, dim=-1)
        out_point_norm = out_point_norm.reshape(*out_point_norm.shape[:2], -1)
        out_point_local = out_point_local.reshape(*out_point_local.shape[:2], -1)

        v_pair = self.pair_value(x2d).reshape(*x2d.shape[:-1], self.n_head, -1)
        out_pair = torch.einsum("bhij,bijhc->bihc", attn, v_pair)
        out_pair = out_pair.reshape(*out_pair.shape[:2], -1)

        out_feat = torch.cat(
            [out_scalar, out_point_local, out_pair, out_point_norm], dim=-1
        )

        x = self.dropout(self.fc_out(out_feat))
        return x


class SAEncoderLayer(nn.Module):
    def __init__(self, d_model, d_pair, n_head, dim_feedforward, dropout):
        super(SAEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SAAttention(
            d_model=d_model, d_pair=d_pair, n_head=n_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x1d, x2d, pose, bias):
        x1d = x1d + self.attn(self.norm1(x1d), x2d, pose, bias)
        x1d = x1d + self.ffn(self.norm2(x1d))
        return x1d


class SAEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(SAEncoder, self).__init__()
        self.layers = nn.ModuleList([SAEncoderLayer(**kwargs) for _ in range(n_layer)])

    def forward(self, x1d, x2d, pose, bias):
        for module in self.layers:
            x1d = module(x1d, x2d, pose, bias)
        return x1d


class StructureModule(nn.Module):
    def __init__(self, d_model, **kwargs):
        super(StructureModule, self).__init__()
        self.encoder = SAEncoder(d_model=d_model, **kwargs)
        self.diff_head = DiffHead(ninp=d_model)

    def forward(self, pose, x1d, x2d, bias):
        T, IR = pose

        x1d = self.encoder(x1d, x2d, pose, bias)

        return self.diff_head(x1d)
