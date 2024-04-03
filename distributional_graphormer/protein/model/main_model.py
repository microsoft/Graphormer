import math

import numpy as np
import torch
import torch.nn.functional as F
from common import config as cfg
from torch import nn

from . import geometry, so3
from .base_model import BaseModel
from .positional_encoding import RelativePositionBias
from .structure_module import StructureModule


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class MainModel(BaseModel):
    def __init__(self, d_model=768, d_pair=256, n_layer=12, n_heads=32):
        super(MainModel, self).__init__()

        self.init_diffusion_params()

        self.step_emb = SinusoidalPositionEmbeddings(dim=d_model)
        self.x1d_proj = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, d_model, bias=False)
        )
        self.x2d_proj = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, d_pair, bias=False)
        )
        self.rp_proj = RelativePositionBias(
            num_buckets=64, max_distance=128, out_dim=d_pair
        )

        self.st_module = StructureModule(
            d_pair=d_pair,
            n_layer=n_layer,
            d_model=d_model,
            n_head=n_heads,
            dim_feedforward=1024,
            dropout=0.1,
        )

    def init_diffusion_params(self):
        self.n_time_step = 500
        self.tr_sigma_min = 0.1
        self.tr_sigma_max = 35
        self.rot_sigma_min = 0.02
        self.rot_sigma_max = 1.65
        self.t_schedule = self._get_t_schedule(self.n_time_step)

    def forward_step(self, input_pose, mask, step, single_repr, pair_repr):
        x1d = self.x1d_proj(single_repr) + self.step_emb(step)[:, None]
        x2d = self.x2d_proj(pair_repr)
        T, IR = input_pose

        pos = torch.arange(T.shape[1], device=x1d.device)
        pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        x2d = x2d + self.rp_proj(pos)[None]

        z = (~mask).long().sum(-1, keepdims=True)
        mask = mask.masked_fill(z == 0, False)

        bias = mask.float().masked_fill(mask, float("-inf"))[:, None, :, None]
        bias = bias.permute(0, 3, 1, 2)

        T_eps, IR_eps = self.st_module((T, IR), x1d, x2d, bias)

        T_eps = torch.matmul(IR.transpose(-1, -2), T_eps.unsqueeze(-1)).squeeze(-1)
        return T_eps, IR_eps

    def _gen_timestep(self, B, device):
        # generate half of the time steps, and the other half are the reverse
        time_step = torch.randint(self.n_time_step, size=(B // 2,)).to(device)
        time_step = torch.cat([time_step, self.n_time_step - 1 - time_step])
        return time_step

    def _get_t_schedule(self, n_time_step):
        return torch.linspace(1, 0, n_time_step + 1)[:-1]

    def _t_to_sigma(self, time_step, device):
        t = self.t_schedule[time_step].to(device)
        T_sigma = (self.tr_sigma_min ** (1 - t)) * (self.tr_sigma_max ** (t))
        IR_sigma = (self.rot_sigma_min ** (1 - t)) * (self.rot_sigma_max ** (t))
        return T_sigma, IR_sigma

    def _gen_noise(self, time_step, T_size, IR_size, device):
        T_sigma, IR_sigma = self._t_to_sigma(time_step, device)  # (B, ), (B, )

        # T_update, T_score
        T_update = torch.stack(
            [
                torch.normal(mean=0, std=T_sigma[i], size=(T_size[1], 3), device=device)
                for i in range(T_size[0])
            ],
            dim=0,
        )
        # T_update: (B, L, 3)
        T_score = -T_update / T_sigma[..., None, None] ** 2

        # generate B x L noises
        def gen_batch_sample(batch, rot_sigma, device):
            eps = rot_sigma.cpu().numpy()
            so3_rot_update_np = so3.batch_sample_vec(batch, eps=eps)
            so3_rot_update = torch.tensor(so3_rot_update_np, device=device)
            so3_rot_mat = geometry.axis_angle_to_matrix(so3_rot_update.squeeze())
            so3_rot_score_np = so3.batch_score_vec(
                batch, vec=so3_rot_update_np, eps=eps
            )
            so3_rot_score = torch.tensor(so3_rot_score_np, device=device)
            so3_rot_score_norm = (
                so3.score_norm(torch.tensor([rot_sigma])).unsqueeze(-1).repeat(batch, 1)
            )

            return so3_rot_update, so3_rot_mat, so3_rot_score, so3_rot_score_norm

        so3_rot_update_stack = []
        so3_rot_mat_stack = []
        so3_rot_score_stack = []
        so3_rot_score_norm_stack = []

        for b in range(IR_size[0]):
            L = IR_size[1]
            rot_sigma = IR_sigma[b]

            (
                so3_rot_update,
                so3_rot_mat,
                so3_rot_score,
                so3_rot_score_norm,
            ) = gen_batch_sample(L, rot_sigma, device)
            so3_rot_update_stack.append(so3_rot_update)
            so3_rot_mat_stack.append(so3_rot_mat)
            so3_rot_score_stack.append(so3_rot_score)
            so3_rot_score_norm_stack.append(so3_rot_score_norm)

        so3_rot_update = torch.stack(so3_rot_update_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3
        )
        so3_rot_mat = torch.stack(so3_rot_mat_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3, 3
        )
        so3_rot_score = torch.stack(so3_rot_score_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3
        )
        rot_score_norm = torch.stack(so3_rot_score_norm_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 1
        )

        return {
            "T_sigma": T_sigma,
            "IR_sigma": IR_sigma,
            "T_update": T_update,
            "T_score": T_score,
            "so3_rot_update": so3_rot_update,
            "so3_rot_mat": so3_rot_mat,
            "so3_rot_score": so3_rot_score,
            "so3_rot_score_norm": rot_score_norm,
        }

    def forward(self, data, compute_loss=True):
        device = data["single_repr"].device
        B, L = data["single_repr"].shape[:2]
        T, IR = data["T"], data["IR"]
        mask = torch.isnan((IR.sum(-1) + T).sum(-1))

        time_step = self._gen_timestep(B, device)
        noise_gen = self._gen_noise(time_step, T.size(), IR.size(), device)

        T_sigma, IR_sigma = noise_gen["T_sigma"], noise_gen["IR_sigma"]
        T_update = noise_gen["T_update"].type_as(T)
        T_score = noise_gen["T_score"].type_as(T)
        noise_gen["so3_rot_update"].type_as(IR)
        so3_rot_mat = noise_gen["so3_rot_mat"].type_as(IR)
        so3_rot_score = noise_gen["so3_rot_score"].type_as(IR)
        so3_rot_score_norm = noise_gen["so3_rot_score_norm"].type_as(IR)

        # modify T and IR with noise
        T_perturbed = T + T_update
        IR_perturbed = torch.matmul(so3_rot_mat, IR)

        T_perturbed.masked_fill_(mask[..., None], 0.0)
        IR_perturbed.masked_fill_(mask[..., None, None], 0.0)
        pred_T_eps, pred_IR_eps = self.forward_step(
            (T_perturbed, IR_perturbed),
            mask,
            time_step,
            data["single_repr"],
            data["pair_repr"],
        )

        target_T_eps = T_score
        target_IR_eps = so3_rot_score

        T_diff_loss = (pred_T_eps - target_T_eps * T_sigma[..., None, None]) ** 2
        IR_diff_loss = (pred_IR_eps - target_IR_eps / so3_rot_score_norm) ** 2

        T_diff_loss.masked_fill_(mask[..., None], 0)
        IR_diff_loss.masked_fill_(mask[..., None], 0)

        loss = 1.0 * T_diff_loss.mean() + 1.0 * IR_diff_loss.mean()

        out = {}
        out["loss"] = loss
        out["T_diff_loss"] = T_diff_loss.mean()
        out["IR_diff_loss"] = IR_diff_loss.mean()
        out["update_loss"] = out["loss"]

        return out
