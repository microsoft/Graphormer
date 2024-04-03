# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.utils import safe_getattr, safe_hasattr
from ..modules import init_graphormer_params
from ..modules import GraphormerGraphEncoder as GraphormerGraphEncoderBase
from ..modules import NodeTaskHead, Graph3DBias
from .oc_pbc import Graph3DBiasPBCCutoff
from .graphormer_encoder import GraphormerEncoder as GraphormerEncoderBase

import pickle as pkl

from scipy import integrate

@torch.jit.script
def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeStepEncoder(nn.Module):
    def __init__(self, d):
        super(TimeStepEncoder, self).__init__()
        self.time_emb = SinusoidalPositionEmbeddings(d)

    def forward(self, x):
        h = self.time_emb(x)
        return h


class CellExpander:
    def __init__(self, cutoff=10.0):
        self.cells = [
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
        self.cutoff = cutoff

    def expand(self, batched_data):
        pos = batched_data["pos"] # B x T x 3
        init_pos = batched_data["init_pos"] # B x T x 3
        atoms = batched_data["x"][:, :, 0]
        batch_size, max_num_atoms = pos.size()[:2]
        cell = batched_data["cell"] # B x 3 x 3
        cell_tensor = torch.tensor(self.cells, device=pos.device).to(cell.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        offset = torch.bmm(cell_tensor, cell) # B x 8 x 3
        expand_pos = pos.unsqueeze(1) + offset.unsqueeze(2) # B x 8 x T x 3
        expand_pos = expand_pos.view(batch_size, -1, 3) # B x (8 x T) x 3
        init_expand_pos = init_pos.unsqueeze(1) + offset.unsqueeze(2) # B x 8 x T x 3
        init_expand_pos = init_expand_pos.view(batch_size, -1, 3) # B x (8 x T) x 3
        expand_dist = torch.norm(pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1) # B x T x (8 x T)
        expand_mask = expand_dist < self.cutoff # B x T x (8 x T)
        expand_mask = torch.masked_fill(expand_mask, atoms.eq(0).unsqueeze(-1), False)
        expand_mask = (torch.sum(expand_mask, dim=1) > 0) & (~(atoms.eq(0).repeat(1, 8))) # B x (8 x T)
        expand_len = torch.sum(expand_mask, dim=-1)
        max_expand_len = torch.max(expand_len)
        outcell_index = torch.zeros([batch_size, max_expand_len], dtype=torch.long, device=pos.device)
        expand_pos_compressed = torch.zeros([batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device)
        init_expand_pos_compressed = torch.zeros([batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device)
        outcell_all_index = torch.arange(max_num_atoms, dtype=torch.long, device=pos.device).repeat(len(self.cells))
        for i in range(batch_size):
           outcell_index[i, :expand_len[i]] = outcell_all_index[expand_mask[i]]
           expand_pos_compressed[i, :expand_len[i], :] = expand_pos[i, expand_mask[i], :]
           init_expand_pos_compressed[i, :expand_len[i], :] = init_expand_pos[i, expand_mask[i], :]
        batched_data["expand_pos"] = expand_pos_compressed
        batched_data["init_expand_pos"] = init_expand_pos_compressed
        batched_data["expand_len"] = expand_len
        batched_data["outcell_index"] = outcell_index
        batched_data["expand_mask"] = mask_after_k_persample(batch_size, max_expand_len, expand_len)

        return batched_data


class GraphormerGraphEncoder(GraphormerGraphEncoderBase):
    """
    User-defined extra node layers or bias layers
    """

    def init_extra_node_layers(self, args):
        super().init_extra_node_layers(args)
        self.timestep_encoder = TimeStepEncoder(args.encoder_embed_dim)
        self.dist_feature_node_scale = args.dist_feature_node_scale
        self.tag_embedding = nn.Embedding(4, args.encoder_embed_dim)
        self.no_diffusion = args.no_diffusion
        self.pbc_approach = args.pbc_approach
        self.use_bonds = args.use_bonds
        pass

    def init_extra_bias_layers(self, args):
        super().init_extra_bias_layers(args)
        pass

    def forward_extra_node_layers(self, batched_data, x):
        x = super().forward_extra_node_layers(batched_data, x)
        assert x.shape[0] == batched_data["ts"].shape[0]
        if not self.no_diffusion:
            ts = batched_data["ts"]  # [B, ]
            ts_emb = self.timestep_encoder(ts).type_as(x)  # [B, d]
            x = x + ts_emb[:, None, :]
        x[:, 1:, :] = x[:, 1:, :] + self._edge_features * self.dist_feature_node_scale

        tags = batched_data["tags"].long()
        tag_features = self.tag_embedding(tags)
        x[:, 1:, :] += tag_features

        self._edge_features = None
        return x

    def forward_extra_bias_layers(self, batched_data, attn_bias):
        bias = super().forward_extra_bias_layers(batched_data, attn_bias)
        if self.pbc_approach == "cutoff":
            max_len = bias.size()[2]
            num_heads = bias.size()[1]
            outcell_index = batched_data["outcell_index"].unsqueeze(1).unsqueeze(2).repeat(1, num_heads, max_len, 1)
            extended_bias = torch.gather(bias[:, :, :, 1:], dim=3, index=outcell_index)
            bias = torch.cat([bias, extended_bias], dim=3)
        if self.use_bonds:
            bias[:, :, 1:, 1:] += self._attn_bias_3d
        else:
            bias[:, :, 1:, 1:] = self._attn_bias_3d
        self._bias = bias
        self._attn_bias_3d = None
        return bias


class GraphormerEncoder(GraphormerEncoderBase):
    def build_graph_encoder(self, args):
        return GraphormerGraphEncoder(args)


def make_masks(batched_data):
    if "lig_mask" not in batched_data:
        n_graphs, n_atoms = batched_data["x"].shape[:2]  # G, T
        lnode = batched_data["lnode"]
        batched_data["lig_mask"] = ~mask_after_k_persample(n_graphs, n_atoms, lnode)


def get_center_pos(batched_data):
    # return ligand center position, return [G, 3]
    make_masks(batched_data)
    lig_mask = batched_data["lig_mask"]
    c = (
        torch.sum(batched_data["pos"] * lig_mask[:, :, None], axis=1).unsqueeze(1)
        / batched_data["lnode"][:, None, None]
    )
    return c


def get_init_center_pos(batched_data):
    # return ligand center position, return [G, 3]
    make_masks(batched_data)
    lig_mask = batched_data["lig_mask"]
    c = (
        torch.sum(batched_data["init_pos"] * lig_mask[:, :, None], axis=1).unsqueeze(1)
        / batched_data["lnode"][:, None, None]
    )
    return c


def sample_time_step(n, num_timesteps, device):
    time_step = torch.randint(0, num_timesteps, size=(n // 2 + 1,), device=device)
    time_step = torch.cat([time_step, num_timesteps - time_step - 1], dim=0)[:n]
    return time_step


def tensor_merge(cond, input, other):
    return cond * input + (~cond) * other


def tensor_merge_truncated(cond, input, other, max_len):
    ret = (~cond) * other
    ret[:, :max_len, :] = cond[:, :max_len, :] * input
    return ret


@dataclass
class GraphormerDiffModelConfig(FairseqDataclass):
    num_diffusion_timesteps: int = field(
        default=5000, metadata={"help": "number of diffusion timesteps"}
    )
    diffusion_beta_schedule: str = field(
        default="sigmoid", metadata={"help": "beta schedule for diffusion"}
    )
    diffusion_beta_start: float = field(
        default=1.0e-7, metadata={"help": "beta start for diffusion"}
    )
    diffusion_beta_end: float = field(
        default=2.0e-3, metadata={"help": "beta end for diffusion"}
    )
    diffusion_sampling: str = field(
        default="ddpm", metadata={"help": "sampling strategy, ddpm or ddim"},
    )
    ddim_steps: int = field(
        default=50, metadata={"help": "sampling steps for ddim"},
    )
    ddim_eta: float = field(
        default=0.0, metadata={"help": "eta for ddim"},
    )
    num_atom_types: int = field(
        default=128, metadata={"help": "number of atom types"},
    )
    dist_feature_extractor: str = field(
        default="rbf", metadata={"help": "distance feature extractor, can be rbf or gbf"},
    )
    dist_feature_node_scale: float = field(
        default=1.0, metadata={"help": "scale of distance feature added to node representations"},
    )
    dist_feature_num_kernels: int = field(
        default=128, metadata={"help": "number of kernels to extract distance features"},
    )
    no_diffusion: bool = field(
        default=False, metadata={"help": "disable diffusion"},
    )
    max_pbc_cell_rep_threshold: int = field(
        default=10, metadata={"help": "number of PBC cell types"},
    )
    max_num_neighbors_threshold: int = field(
        default=10, metadata={"help": "maximum number of neighbors in PBC edge graph"},
    )
    pbc_approach: str = field(
        default="none", metadata={"help": "PBC for graph construction, can be oc_pbc, cutoff, or none"},
    )
    pbc_cutoff: float = field(
        default=6.0, metadata={"help": "cutoff for PBC"},
    )
    diffusion_noise_std: float = field(
        default=1.0, metadata={"help": "noise std for diffusion"},
    )
    remove_head: bool = field(
        default=False, metadata={"help": "whether to remove force_proj head parameters when loading checkpoints"},
    )
    use_bonds: bool = field(
        default=False, metadata={"help": "whether to use bonds in graph"},
    )
    num_epsilon_estimator: int = field(
        default=10, metadata={"help": "number of epsilons to sampled for trace estimation in flow ode"},
    )


@register_model("graphormer_diff", dataclass=GraphormerDiffModelConfig)
class GraphormerDiffModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)

        self.init_diffusion(args)

        self.pbc_approach = args.pbc_approach
        if self.pbc_approach == "cutoff":
            self.graph_3d_bias = Graph3DBiasPBCCutoff(
                num_heads=args.encoder_attention_heads,
                num_atom_types=args.num_atom_types,
                num_layers=args.encoder_layers,
                embed_dim=args.encoder_embed_dim,
                num_kernel=args.dist_feature_num_kernels,
                dist_feature_extractor=args.dist_feature_extractor,
                num_diffusion_timesteps=args.num_diffusion_timesteps,
                no_share_rpe=False
            )
        elif self.pbc_approach == "none":
            self.graph_3d_bias = Graph3DBias(
                num_heads=args.encoder_attention_heads,
                num_atom_types=args.num_atom_types,
                num_layers=args.encoder_layers,
                embed_dim=args.encoder_embed_dim,
                num_kernel=args.dist_feature_num_kernels,
                dist_feature_extractor=args.dist_feature_extractor,
                num_diffusion_timesteps=args.num_diffusion_timesteps,
                no_share_rpe=False
            )
        else:
            raise ValueError(f"Unknown PBC approach {self.pbc_approach}")

        self.force_proj = NodeTaskHead(
            args.encoder_embed_dim, args.encoder_attention_heads
        )

        self.force_proj_delta_pos = NodeTaskHead(
            args.encoder_embed_dim, args.encoder_attention_heads
        )

        self.diffusion_sampling = self.args.diffusion_sampling
        self.ddim_steps = self.args.ddim_steps
        self.ddim_eta = self.args.ddim_eta
        self.no_diffusion = self.args.no_diffusion
        self.diffusion_noise_std = self.args.diffusion_noise_std
        self.remove_head = self.args.remove_head
        self.use_bonds = self.args.use_bonds
        self.num_epsilon_estimator = self.args.num_epsilon_estimator

        self.cell_expander = CellExpander(self.args.pbc_cutoff)

        print(f"{self.__class__.__name__}: {self}")

    def max_nodes(self):
        return self.encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        encoder = GraphormerEncoder(args)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        if self.pbc_approach == "cutoff":
            batched_data = self.cell_expander.expand(batched_data)
        n_atoms = batched_data["x"].shape[1]
        graph_3d = self.graph_3d_bias(batched_data)
        delta_pos = graph_3d["delta_pos"]
        outcell_index = batched_data.get("outcell_index", None)

        # (G, H, T, T)
        self.encoder.graph_encoder._attn_bias_3d = graph_3d["attn_bias_3d"]
        # (G, T, D)
        self.encoder.graph_encoder._edge_features = graph_3d["edge_features"]
        encoder_out = self.encoder(batched_data, **kwargs)
        x = encoder_out["encoder_out"][0].transpose(0, 1)  # B x T x D
        attn_bias = self.encoder.graph_encoder._bias

        extend_n_atoms = attn_bias.size()[-1] - 1

        force = self.force_proj(
            x[:, 1:], attn_bias[:, :, 1:, 1:].reshape(-1, n_atoms, extend_n_atoms), delta_pos, outcell_index
        )

        force_delta_pos = self.force_proj_delta_pos(
            x[:, 1:], attn_bias[:, :, 1:, 1:].reshape(-1, n_atoms, extend_n_atoms), delta_pos, outcell_index
        )

        # to avoid memory leak
        self.encoder.graph_encoder._attn_bias_3d = None
        self.encoder.graph_encoder._edge_features = None
        self.encoder.graph_encoder._bias = None

        # trick, use force as delta_pos prediction for loading pretrained is2rs model
        return force_delta_pos, force

    def init_diffusion(self, args):
        num_diffusion_timesteps = args.num_diffusion_timesteps

        betas = get_beta_schedule(
            beta_schedule=args.diffusion_beta_schedule,
            beta_start=args.diffusion_beta_start,
            beta_end=args.diffusion_beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        ## variances
        alphas = (1.0 - betas).cumprod(dim=0)
        num_timesteps = betas.size(0)

        self.num_timesteps = num_timesteps
        self.alphas = alphas
        self.betas = betas

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        # some paramters need to keep fp32
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        return self

    def get_sampling_output(self, batched_data, pos_center=None, **kwargs):
        make_masks(batched_data)
        device = batched_data["x"].device

        n_graphs, n_atoms = batched_data["x"].shape[:2]
        lig_mask = batched_data["lig_mask"]

        orig_pos = batched_data["pos"]
        init_pos = batched_data["init_pos"]

        # centering
        if pos_center is None:
            # use ligand center position
            pos_center = get_init_center_pos(batched_data)

        orig_pos -= pos_center
        init_pos -= pos_center

        delta_pos = orig_pos - init_pos

        # fill ligand pos with noise, keep protein pos
        pos_noise = torch.zeros(size=orig_pos.size(), device=device)
        if not self.no_diffusion:
            pos_noise = pos_noise.normal_() * self.diffusion_noise_std
        delta_pos = tensor_merge(lig_mask[:, :, None], pos_noise, delta_pos)
        orig_delta_pos = delta_pos.clone()

        # protein pos and noisy ligand pos
        batched_data["pos"] = init_pos + delta_pos

        if not self.no_diffusion:
            if self.args.diffusion_sampling == "ddpm":
                # Sampling from Step T-1 to Step 0
                for t in range(self.num_timesteps - 1, -1, -1):
                    hat_alpha_t = self.alphas[t]
                    hat_alpha_t_1 = 1.0 if t == 0 else self.alphas[t - 1]
                    alpha_t = hat_alpha_t / hat_alpha_t_1
                    beta_t = 1 - alpha_t
                    sigma_t = (
                        0.0
                        if t == 0
                        else ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt()
                    )

                    # forward
                    batched_data["ts"] = torch.ones(n_graphs, device=device, dtype=torch.long).fill_(t)
                    force, force_delta_pos = self(batched_data, **kwargs)
                    force = force.detach()
                    force_delta_pos = force_delta_pos.detach()

                    epsilon = torch.zeros_like(batched_data["pos"]).normal_() * self.diffusion_noise_std

                    delta_pos = batched_data["pos"] - batched_data["init_pos"]
                    lig_pos = (
                        delta_pos - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * force
                    ) / alpha_t.sqrt() + sigma_t * epsilon

                    # update positions (only ligand)
                    batched_data["pos"] = tensor_merge(
                        lig_mask[:, :, None], lig_pos, delta_pos
                    ) + batched_data["init_pos"]
                    batched_data["pos"] = batched_data["pos"].detach()
            elif self.diffusion_sampling == "ddim":
                sampled_steps, _ = torch.sort((torch.randperm(self.num_timesteps - 2, dtype=torch.long, device=device) + 1)[:self.ddim_steps - 1])
                sampled_steps = torch.cat([
                    sampled_steps,
                    torch.tensor([self.num_timesteps - 1], device=device).long()])
                for i in range(sampled_steps.shape[0] - 1, 0, -1):
                    t = sampled_steps[i]
                    t_1 = sampled_steps[i - 1]
                    hat_alpha_t = self.alphas[t]
                    hat_alpha_t_1 = self.alphas[t_1]
                    alpha_t = hat_alpha_t / hat_alpha_t_1
                    beta_t = 1.0 - alpha_t
                    sigma_t = self.ddim_eta * ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt()

                    # forward
                    batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long).fill_(t)
                    force, force_delta_pos = self.forward(batched_data, sampling=False, original_pos=None, **kwargs)
                    lig_pos = batched_data["pos"] - batched_data["init_pos"]
                    x_0_pred = (lig_pos - (1.0 - hat_alpha_t).sqrt() * force) / hat_alpha_t.sqrt()
                    epsilon = torch.zeros_like(lig_pos).normal_() * self.diffusion_noise_std
                    lig_pos = hat_alpha_t_1.sqrt() * x_0_pred +\
                        (1.0 - hat_alpha_t_1 - sigma_t ** 2).sqrt() * (lig_pos - hat_alpha_t.sqrt() * x_0_pred) / (1.0 - hat_alpha_t).sqrt() + sigma_t * epsilon
                    batched_data["pos"] = tensor_merge(lig_mask[:, :, None], lig_pos + batched_data["init_pos"], batched_data["pos"])
                    batched_data["pos"] = batched_data["pos"].detach()

                # forward for last step
                t = sampled_steps[0]
                hat_alpha_t = self.alphas[t]

                # forward
                batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long).fill_(t)
                force, force_delta_pos = self.forward(batched_data, sampling=False, original_pos=None, **kwargs)
                lig_pos = batched_data["pos"] - batched_data["init_pos"]
                x_0_pred = (lig_pos - (1.0 - hat_alpha_t).sqrt() * force) / hat_alpha_t.sqrt()
                batched_data["pos"] = tensor_merge(lig_mask[:, :, None], x_0_pred + batched_data["init_pos"], batched_data["pos"])
                batched_data["pos"] = batched_data["pos"].detach()
            else:
                raise ValueError(f"Unknown diffusion sampling strategy {self.args.diffusion_sampling}. Support only ddim and ddpm.")
        else:
            batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long)
            _, lig_pos = self.forward(batched_data, **kwargs)
            lig_pos = lig_pos.detach()
            batched_data["pos"] = tensor_merge(lig_mask[:, :, None], lig_pos + init_pos, batched_data["pos"])

        pred_pos = batched_data["pos"].clone()

        batched_data["init_pos"] += pos_center

        loss = (pred_pos - orig_pos) ** 2
        loss = torch.sum(loss, dim=-1, keepdim=True)

        if not self.no_diffusion:
            delta_pos_loss = (force_delta_pos - orig_delta_pos) ** 2
            delta_pos_loss = torch.sum(delta_pos_loss, dim=-1, keepdim=True)

        # only account for ligand atoms
        loss = loss.masked_fill_(~lig_mask[:, :, None], 0)
        if not self.no_diffusion:
            delta_pos_loss = delta_pos_loss.masked_fill_(~lig_mask[:, :, None], 0)

        rmsd = torch.sqrt(torch.sum(loss, dim=-2) / batched_data["lnode"][:, None])
        if not self.no_diffusion:
            delta_pos_rmsd = torch.sqrt(torch.sum(delta_pos_loss, dim=-2) / batched_data["lnode"][:, None])

        return {
            "pred_pos": pred_pos + pos_center,
            "persample_loss": loss,
            "persample_rmsd": rmsd,
            "persample_delta_pos_rmsd": delta_pos_rmsd if not self.no_diffusion else torch.zeros_like(rmsd),
            "sample_size": n_graphs,
        }

    def get_training_output(self, batched_data, **kwargs):
        make_masks(batched_data)
        device = batched_data["x"].device
        n_graphs, _ = batched_data["x"].shape[:2]  # G, T
        # paddings:  (G, T)
        lig_mask = batched_data["lig_mask"]

        # postions centering (center of movable atoms in initial positions)
        init_center = get_init_center_pos(batched_data)
        batched_data["pos"] -= init_center
        batched_data["init_pos"] -= init_center
        delta_pos = batched_data["pos"] - batched_data["init_pos"]

        pos = batched_data["pos"]
        if self.no_diffusion:
            orig_pos = delta_pos * lig_mask[:, :, None]

        if not self.no_diffusion:
            time_step = sample_time_step(n_graphs, self.num_timesteps, device)
            a = self.alphas.index_select(0, time_step)  # (G, )

            # Perturb pos, only on ligand atoms
            a_pos = a[:, None, None]  # (G, 1, 1)
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        if not self.no_diffusion:
            pos_noise = pos_noise.normal_() * self.diffusion_noise_std
        pos_noise = pos_noise.masked_fill((~lig_mask)[:, :, None], 0)

        if not self.no_diffusion:
            a_pos_scale = a_pos.sqrt().masked_fill((~lig_mask)[:, :, None], 1.0)
            pos_perturbed = batched_data["init_pos"] + delta_pos * a_pos_scale + pos_noise * (1.0 - a_pos).sqrt()
        else:
            pos_perturbed = tensor_merge(lig_mask.unsqueeze(-1), pos_noise + batched_data["init_pos"], pos)
        batched_data["pos"] = pos_perturbed.type_as(self.alphas)

        if not self.no_diffusion:
            batched_data["ts"] = time_step
        else:
            batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long)

        # forward
        pred_eps, delta_pos_pred_eps = self(batched_data, **kwargs)

        if not self.no_diffusion:
            target_eps = pos_noise
            delta_pos_target_eps = delta_pos
        else:
            pred_eps = delta_pos_pred_eps
            target_eps = orig_pos

        # calculate loss
        loss = (pred_eps - target_eps) ** 2
        loss = torch.sum(loss, dim=-1, keepdim=True)
        if not self.no_diffusion:
            delta_pos_loss = (delta_pos_pred_eps - delta_pos_target_eps) ** 2
            delta_pos_loss = torch.sum(delta_pos_loss, dim=-1, keepdim=True)

        # only account for ligand atoms
        loss = loss.masked_fill((~lig_mask)[:, :, None], 0.0)
        if not self.no_diffusion:
            delta_pos_loss = delta_pos_loss.masked_fill((~lig_mask)[:, :, None], 0.0)
            delta_pos_rmsd = torch.sqrt(torch.sum(delta_pos_loss, dim=1) / batched_data["lnode"].unsqueeze(1))
        else:
            delta_pos_loss = torch.zeros_like(loss)
            delta_pos_rmsd = torch.sqrt(torch.sum(delta_pos_loss, dim=1) / batched_data["lnode"].unsqueeze(1))

        return {
            "persample_loss": loss,
            "persample_diffusion_loss": loss,
            "persample_delta_pos_loss": delta_pos_loss,
            "persample_delta_pos_rmsd": delta_pos_rmsd,
            "sample_size": n_graphs,
        }


    def get_flow_ode_output(self, batched_data, offset=None, **kwargs):
        make_masks(batched_data)
        n_graphs = batched_data["x"].size()[0]
        device = batched_data["x"].device
        lig_mask = batched_data["lig_mask"]
        init_pos_clone = batched_data["init_pos"].clone()

        if offset is not None:
            ads_mask = (batched_data["tags"] == 3).unsqueeze(-1)
            cell = batched_data["cell"].transpose(1, 2).float() # B x 3 x 3
            offseted_init_pos = tensor_merge(ads_mask, batched_data["init_pos"] + offset.unsqueeze(1), batched_data["init_pos"]).transpose(1, 2).float() # B x 3 x T
            solv = torch.linalg.solve(cell, offseted_init_pos).transpose(1, 2) # B x T x 3
            solv[:, :, 0] %= 1.0
            solv[:, :, 0] %= 1.0
            solv[:, :, 1] %= 1.0
            solv[:, :, 1] %= 1.0
            offseted_init_pos = torch.matmul(solv, cell.transpose(1, 2))
            batched_data["init_pos"] = tensor_merge(ads_mask, offseted_init_pos, init_pos_clone)

        init_center = get_init_center_pos(batched_data)
        batched_data["pos"] -= init_center
        batched_data["init_pos"] -= init_center
        batched_data["pred_pos"] -= init_center

        max_num_moveable = torch.max(batched_data["num_moveable"])
        index = torch.arange(n_graphs, device=device)
        trace_index = torch.arange(3 * max_num_moveable, device=device)

        def score_fn(x_pos, batched_data, t, max_num_moveable):
            pos_clone = batched_data["pos"].clone()
            batched_data["ts"] = torch.ones(n_graphs, device=device, dtype=torch.long).fill_(t)
            batched_data["pos"] = tensor_merge_truncated(lig_mask.unsqueeze(-1), x_pos, pos_clone, max_num_moveable)
            a = self.alphas[t]
            ret, _ = self(batched_data, **kwargs)
            ret = -1.0 * ret / (1.0 - a).sqrt()
            batched_data["pos"] = pos_clone
            return ret[:, :max_num_moveable, :]

        def ode_fn(t, pos_and_p_numpy):
            with torch.set_grad_enabled(True) and torch.autograd.set_detect_anomaly(True):
                beta = self.betas[t]
                pos_and_p = pos_and_p_numpy.clone()
                pos = torch.nn.parameter.Parameter(pos_and_p[:-n_graphs].reshape(n_graphs, max_num_moveable, 3).clone(), requires_grad=True)

                if self.num_epsilon_estimator <= 0:
                    reverse_flow_ode_fn = lambda pos: (-0.5 * beta * score_fn(pos, batched_data, t, max_num_moveable) - \
                        (1.0 - (1.0 - beta).sqrt()) * (pos - batched_data["init_pos"][:, :max_num_moveable])).\
                        masked_fill((~lig_mask[:, :max_num_moveable]).unsqueeze(-1), 0.0)
                    reverse_flow_ode = reverse_flow_ode_fn(pos)
                    jacob = torch.autograd.functional.jacobian(reverse_flow_ode_fn, pos)
                    jacob = jacob.reshape(n_graphs, max_num_moveable * 3, n_graphs, max_num_moveable, 3)
                    jacob = jacob.transpose(1, 2).view(n_graphs, n_graphs, max_num_moveable, 3, max_num_moveable, 3)
                    jacob = jacob[index, index]
                    jacob = jacob.reshape(n_graphs, 3 * max_num_moveable, 3 * max_num_moveable)[:, trace_index, trace_index]
                    trace = torch.sum(jacob, dim=-1)
                else:
                    estimated_traces = []
                    for_estimator = -0.5 * beta * score_fn(pos, batched_data, t, max_num_moveable)
                    reverse_flow_ode = (for_estimator - (1.0 - (1.0 - beta).sqrt()) * (pos - batched_data["init_pos"][:, :max_num_moveable]))
                    for_estimator = for_estimator.masked_fill((~lig_mask[:, :max_num_moveable]).unsqueeze(-1), 0.0)
                    reverse_flow_ode = reverse_flow_ode.masked_fill((~lig_mask[:, :max_num_moveable]).unsqueeze(-1), 0.0)
                    for _ in range(self.num_epsilon_estimator):
                        epsilon = torch.zeros_like(batched_data["pos"][:, :max_num_moveable, :]).normal_()
                        drift_prod_sum = torch.sum(torch.sum(epsilon * for_estimator, dim=[1, 2]))
                        grad = torch.autograd.grad(drift_prod_sum, pos, retain_graph=True)[0]
                        grad = grad.masked_fill((~lig_mask[:, :max_num_moveable]).unsqueeze(-1), 0.0)
                        estimated_trace = torch.sum(grad * epsilon, dim=[1, 2]) - (1.0 - (1.0 - beta).sqrt()) * torch.sum(lig_mask[:, :max_num_moveable] * 3.0, dim=1)
                        estimated_traces.append(estimated_trace.unsqueeze(0))
                    estimated_traces = torch.cat(estimated_traces, dim=0)
                    trace = torch.mean(estimated_traces, dim=0)
            ret = torch.cat([reverse_flow_ode.reshape(-1), trace], dim=0).detach()
            return ret, for_estimator.detach()

        init = torch.cat([batched_data["pred_pos"][:, :max_num_moveable, :].reshape(-1), torch.zeros(n_graphs, dtype=torch.float64, device=device)], dim=0)#.cpu()
        for_estimator_sum = torch.zeros_like(batched_data["pred_pos"][:, :max_num_moveable])
        pos_and_p_numpy = init
        for t in range(self.num_timesteps):
            delta_pos_and_p_numpy, delta_for_estimator = ode_fn(t, pos_and_p_numpy)
            pos_and_p_numpy += delta_pos_and_p_numpy
            for_estimator_sum += delta_for_estimator
        sol = pos_and_p_numpy
        likelihood = sol[-n_graphs:]
        latent_pos = (sol[:-n_graphs].reshape(n_graphs, max_num_moveable, 3) - batched_data["init_pos"][:, :max_num_moveable, :])
        lig_mask = lig_mask.unsqueeze(-1)
        latent_pos = latent_pos * lig_mask[:, :max_num_moveable, :]
        prior_log_p = -0.5 * torch.sum(lig_mask * 3.0, dim=[1, 2]) * np.log(2 * np.pi) - torch.sum((latent_pos ** 2) / 2.0, dim=[1, 2])
        likelihood += prior_log_p

        batched_data["pos"] += init_center
        batched_data["init_pos"] += init_center
        batched_data["pred_pos"] += init_center

        return {
            "persample_likelihood": likelihood,
            "persample_prior_log_p": prior_log_p,
            "latent_pos": latent_pos,
            "sample_size": n_graphs,
        }


    def upgrade_state_dict(self, state_dict):
        state_dict_keys = list(state_dict.keys())

        for key in state_dict_keys:
            if key == "force_proj.force_proj.bias":
                state_dict.pop(key)
            elif key == "force_proj_delta_pos.force_proj.bias":
                state_dict.pop(key)

        for key in state_dict_keys:
            if key.find("force_proj_delta_pos") != -1: # this is the latest architecture
                print("force_proj_delta_pos found")
                return

        named_parameters = {
            k: v
            for k, v in self.named_parameters()
        }

        collect_keys_1 = {}
        collect_keys_2 = {}
        collect_keys_3 = {}
        for key in state_dict_keys:
            if key.find("force_proj1") != -1:
                collect_keys_1[key] = state_dict[key].data.clone()
            elif key.find("force_proj2") != -1:
                collect_keys_2[key] = state_dict[key].data.clone()
            elif key.find("force_proj3") != -1:
                collect_keys_3[key] = state_dict[key].data.clone()

        mean_weight_dict = {}
        for key in collect_keys_1:
            weight_sum = collect_keys_1[key].clone()
            key_2 = key.replace("1", "2")
            weight_sum += collect_keys_2[key_2]
            key_3 = key.replace("1", "3")
            weight_sum += collect_keys_3[key_3]
            weight_sum /= 3.0
            mean_key = key.replace("1", "")
            mean_weight_dict[mean_key] = weight_sum

        for key in state_dict_keys:
            if (key.find("force_proj1") != -1 or key.find("force_proj2") != -1 or key.find("force_proj3") != -1):
                state_dict.pop(key)

        for key in mean_weight_dict:
            state_dict[key] = mean_weight_dict[key].clone()

        if not self.remove_head:
            for key in named_parameters:
                if key.find("force_proj_delta_pos") != -1:
                    copy_key = key.replace("force_proj_delta_pos", "force_proj")
                    if copy_key.find("1") != -1:
                        copy_key = copy_key.replace("1", "")
                        state_dict[key] = mean_weight_dict[copy_key].data.clone()
                    elif copy_key.find("2") != -1:
                        copy_key = copy_key.replace("2", "")
                        state_dict[key] = mean_weight_dict[copy_key].data.clone()
                    elif copy_key.find("3") != -1:
                        copy_key = copy_key.replace("3", "")
                        state_dict[key] = mean_weight_dict[copy_key].data.clone()
                    else:
                        state_dict[key] = state_dict[copy_key].data.clone()
                    print("Copying", key, f"(from {copy_key})")
                elif key.find("force_proj.force_proj") != -1:
                    copy_key = key
                    if copy_key.find("1") != -1:
                        copy_key = copy_key.replace("1", "")
                        state_dict[key] = mean_weight_dict[copy_key].data.clone()
                    elif copy_key.find("2") != -1:
                        copy_key = copy_key.replace("2", "")
                        state_dict[key] = mean_weight_dict[copy_key].data.clone()
                    elif copy_key.find("3") != -1:
                        copy_key = copy_key.replace("3", "")
                        state_dict[key] = mean_weight_dict[copy_key].data.clone()
                    print("Copying", key, f"(from {copy_key})")
        else:
            for key in state_dict_keys:
                init_val = named_parameters[key].data
                if key.find("force_proj") != -1:
                    state_dict[key] = init_val
                    print("Copying", key, "(from init value)")


@register_model_architecture("graphormer_diff", "graphormer_diff_base")
def base_architecture(args):
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.act_dropout = safe_getattr(args, "act_dropout", 0.0)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 768)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = safe_getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = safe_getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # embeddings
    args.num_atoms = safe_getattr(args, "num_atoms", 512 * 9)
    args.num_edges = safe_getattr(args, "num_edges", 512 * 3)
    args.num_in_degree = safe_getattr(args, "num_in_degree", 512)
    args.num_out_degree = safe_getattr(args, "num_out_degree", 512)
    args.num_spatial = safe_getattr(args, "num_spatial", 513)
    args.num_edge_dis = safe_getattr(args, "num_edge_dis", 128)
    args.multi_hop_max_dist = safe_getattr(args, "multi_hop_max_dist", 5)
    args.edge_type = safe_getattr(args, "edge_type", "multi_hop")

    # tricks, disabled by default
    args.layer_scale = safe_getattr(args, "layer_scale", 0.0)
    args.droppath_prob = safe_getattr(args, "droppath_prob", 0.0)
    args.sandwich_norm = safe_getattr(args, "sandwich_norm", False)

    args.num_diffusion_timesteps = safe_getattr(args, "num_diffusion_timesteps", 5000)
    args.diffusion_beta_schedule = safe_getattr(args, "diffusion_beta_schedule", "sigmoid")
    args.diffusion_beta_start = safe_getattr(args, "diffusion_beta_start", 1.0e-7)
    args.diffusion_beta_end = safe_getattr(args, "diffusion_beta_end", 2e-3)
    args.diffusion_sampling = safe_getattr(args, "diffusion_sampling", "ddpm")
    args.ddim_steps = safe_getattr(args, "ddim_steps", 50)
    args.ddim_eta = safe_getattr(args, "ddim_eta", 0.0)
    args.num_atom_types = safe_getattr(args, "num_atom_types", 256)
    args.dist_feature_extractor = safe_getattr(args, "dist_feature_extractor", "rbf")
    args.dist_feature_node_scale = safe_getattr(args, "dist_feature_node_scale", 1.0)
    args.dist_feature_num_kernels = safe_getattr(args, "dist_feature_num_kernels", 128)
    args.no_diffusion = safe_getattr(args, "no_diffusion", False)
    args.max_pbc_cell_rep_threshold = safe_getattr(args, "max_pbc_cell_rep_threshold", 10)
    args.max_num_neighbors_threshold = safe_getattr(args, "max_num_neighbors_threshold", 10)
    args.pbc_approach = safe_getattr(args, "pbc_approach", "none")
    args.pbc_cutoff = safe_getattr(args, "pbc_cutoff", 6.0)
    args.diffusion_noise_std = safe_getattr(args, "diffusion_noise_std", 1.0)
    args.remove_head = safe_getattr(args, "remove_head", False)
    args.use_bonds = safe_getattr(args, "use_bonds", False)
    args.num_epsilon_estimator = safe_getattr(args, "num_epsilon_estimator", 10)


@register_model_architecture("graphormer_diff", "graphormer_diff_xs")
def small_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 64)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 64)
    base_architecture(args)


@register_model_architecture("graphormer_diff", "graphormer_diff_large")
def large_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1024)
    base_architecture(args)
