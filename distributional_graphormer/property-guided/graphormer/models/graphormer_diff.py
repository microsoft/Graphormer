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
from .oc_pbc import Graph3DBiasPBC, Graph3DBiasPBCCutoff
from .graphormer_encoder import GraphormerEncoder as GraphormerEncoderBase
from ..modules import EquivariantMultiHeadAttention, EquivariantLayerNorm, Distance, EquivariantVectorOutput, ExpNormalSmearing

from .m3gnet_utils import get_bandgap_and_derivative_batched_multi_intervals

import pickle as pkl

from scipy import integrate

from tqdm import tqdm

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


def rmsd_pbc(batched_pos1, batched_pos2, cell, lig_mask, num_moveable):
    delta_pos = (batched_pos2 - batched_pos1).float() # B x T x 3
    cell = cell.float() # B x 3 x 3
    delta_pos_solve = torch.linalg.solve(cell.transpose(-1, -2), delta_pos.transpose(-1, -2)).transpose(-1, -2)
    delta_pos_solve[:, :, 0] %= 1.0
    delta_pos_solve[:, :, 0] %= 1.0
    delta_pos_solve[:, :, 1] %= 1.0
    delta_pos_solve[:, :, 1] %= 1.0
    delta_pos_solve[delta_pos_solve > 0.5] -= 1.0
    min_delta_pos = torch.matmul(delta_pos_solve, cell)
    rmsds = torch.sqrt(torch.sum(torch.sum(min_delta_pos ** 2, dim=-1) * lig_mask, dim=-1) / num_moveable)
    return rmsds


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
            [0, 0, 1],
            [-1, -1, 1],
            [-1, 0, 1],
            [-1, 1, 1],
            [0, -1, 1],
            [0, 1, 1],
            [1, -1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, -1],
            [-1, -1, -1],
            [-1, 0, -1],
            [-1, 1, -1],
            [0, -1, -1],
            [0, 1, -1],
            [1, -1, -1],
            [1, 0, -1],
            [1, 1, -1],
        ]
        self.cutoff = cutoff

    def expand(self, batched_data):
        pos = batched_data["pos"] # B x T x 3
        init_pos = batched_data["init_cell_pos"] # B x T x 3
        atoms = batched_data["x"][:, :, 0]
        batch_size, max_num_atoms = pos.size()[:2]
        cell = batched_data["cell_pred"] # B x 3 x 3
        cell_tensor = torch.tensor(self.cells, device=pos.device).to(cell.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        offset = torch.bmm(cell_tensor, cell) # B x 8 x 3
        expand_pos = pos.unsqueeze(1) + offset.unsqueeze(2) # B x 8 x T x 3
        expand_pos = expand_pos.view(batch_size, -1, 3) # B x (8 x T) x 3
        init_expand_pos = init_pos.unsqueeze(1) + offset.unsqueeze(2) # B x 8 x T x 3
        init_expand_pos = init_expand_pos.view(batch_size, -1, 3) # B x (8 x T) x 3
        expand_dist = torch.norm(pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1) # B x T x (8 x T)
        expand_mask = ((expand_dist < self.cutoff) & (expand_dist > 1e-5)) # B x T x (8 x T)
        expand_mask = torch.masked_fill(expand_mask, atoms.eq(0).unsqueeze(-1), False)
        expand_mask = (torch.sum(expand_mask, dim=1) > 0) & (~(atoms.eq(0).repeat(1, len(self.cells)))) # B x (8 x T)
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


class SpatialEmbedding(nn.Module):
    def __init__(self, embed_dim=1024, max_num_tokens=64):
        super(SpatialEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_num_tokens, embed_dim)

    def forward(self, batched_data):
        z_pos = batched_data["pos_clone"][:, :, -1]
        n_graphs, n_tokens = batched_data["x"].size()[:2]
        device = batched_data["x"].device
        used_mask = batched_data["x"][:, :, 0].eq(0)
        z_pos = z_pos.masked_fill(~used_mask, np.inf)
        _, sorted_indices = torch.sort(z_pos, dim=-1)
        poses = torch.arange(n_tokens, device=device).repeat([n_graphs, 1])
        sorted_pos = torch.gather(poses, dim=-1, index=sorted_indices)
        return self.embedding(sorted_pos), sorted_pos


class GraphormerGraphEncoder(GraphormerGraphEncoderBase):
    """
    User-defined extra node layers or bias layers
    """

    def init_extra_node_layers(self, args):
        super().init_extra_node_layers(args)
        self.timestep_encoder = TimeStepEncoder(args.encoder_embed_dim)
        self.sid_encoder = nn.Embedding(20000, args.encoder_embed_dim)
        self.dist_feature_node_scale = args.dist_feature_node_scale
        self.tag_embedding = nn.Embedding(15, args.encoder_embed_dim)
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

        #sid_embedding = self.sid_encoder(batched_data["sid"]).unsqueeze(1)

        #x += sid_embedding

        tags = batched_data["tags"].long()
        tag_features = self.tag_embedding(tags)
        x[:, 1:, :] += tag_features

        #x[:, 1:, :] += self._spat_embed

        self._edge_features = None
        #self._spat_embed = None
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
        lnode = batched_data["natoms"]
        batched_data["lig_mask"] = ~mask_after_k_persample(n_graphs, n_atoms, lnode)
        batched_data["lig_mask_loss"] = ~mask_after_k_persample(n_graphs, n_atoms, lnode)
        batched_data["lig_mask_loss"][:, batched_data["lnodes"] + 3] = False
        batched_data["lig_mask_loss"][:, batched_data["lnodes"] + 5] = False
        batched_data["lig_mask_loss"][:, batched_data["lnodes"] + 6] = False
        batched_data["lig_mask_loss"][:, batched_data["lnodes"] + 7] = False


def get_center_pos(batched_data):
    # return ligand center position, return [G, 3]
    make_masks(batched_data)
    # get center of cell positions
    center = torch.sum(batched_data["cell"], dim=1, keepdim=True) / 2.0
    return center


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
    lattice_size: float = field(
        default=4.0, metadata={"help": "size of lattice"},
    )
    conditioned_ode_factor: float = field(
        default=0.0, metadata={"help": "factor for conditioned generation"},
    )
    target_bandgap_interval: int = field(
        default=3, metadata={"help": "target bandgap interval"},
    )
    target_bandgap_softmax_temperature: float = field(
        default=5.0, metadata={"help": "temperature for target bandgap softmax"},
    )
    sampling_result_dir: str = field(
        default="checkpoints", metadata={"help": "directory to save sampling results"},
    )
    gpu_device_id_record: int = field(
        default=0, metadata={"help": "gpu device index"},
    )
    seed_record: int = field(
        default=0, metadata={"help": "seed record"},
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
        if self.pbc_approach == "oc_pbc":
            self.graph_3d_bias = Graph3DBiasPBC(
                num_heads=args.encoder_attention_heads,
                num_atom_types=args.num_atom_types,
                num_layers=args.encoder_layers,
                embed_dim=args.encoder_embed_dim,
                num_kernel=args.dist_feature_num_kernels,
                dist_feature_extractor=args.dist_feature_extractor,
                num_diffusion_timesteps=args.num_diffusion_timesteps,
                max_pbc_cell_rep_threshold=args.max_pbc_cell_rep_threshold,
                pbc_radius=args.pbc_cutoff,
                max_num_neighbors_threshold=args.max_num_neighbors_threshold,
                no_share_rpe=False
            )
        elif self.pbc_approach == "cutoff":
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

        self.num_radius_classes = 50
        self.target_radius = 14.0

        #self.radius_proj: Callable[[Tensor], Tensor] = nn.Linear(args.encoder_embed_dim, self.num_radius_classes)

        self.diffusion_sampling = self.args.diffusion_sampling
        self.ddim_steps = self.args.ddim_steps
        self.ddim_eta = self.args.ddim_eta
        self.no_diffusion = self.args.no_diffusion
        self.diffusion_noise_std = self.args.diffusion_noise_std
        self.remove_head = self.args.remove_head
        self.use_bonds = self.args.use_bonds
        self.num_epsilon_estimator = self.args.num_epsilon_estimator
        self.lattice_size = self.args.lattice_size
        self.conditioned_ode_factor = self.args.conditioned_ode_factor
        self.target_bandgap_interval = self.args.target_bandgap_interval
        self.target_bandgap_softmax_temperature = self.args.target_bandgap_softmax_temperature
        self.sampling_result_dir = self.args.sampling_result_dir
        self.gpu_device_id_record = self.args.gpu_device_id_record
        self.seed_record = self.args.seed_record

        self.output_model_noise = EquivariantVectorOutput(args.encoder_embed_dim)
        self.out_norm_vec = EquivariantLayerNorm(args.encoder_embed_dim)
        self.distance = Distance()
        self.distance_expansion = ExpNormalSmearing(num_rbf=args.dist_feature_num_kernels)
        self.out_norm = nn.LayerNorm(args.encoder_embed_dim)
        self.attention_layers = nn.ModuleList()
        for _ in range(4):
            layer = EquivariantMultiHeadAttention(
                hidden_channels=args.encoder_embed_dim,
                num_rbf=args.dist_feature_num_kernels,
                num_heads=args.encoder_attention_heads,
            )
            # layer = EquivariantMultiHeadAttention()
            self.attention_layers.append(layer)

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

        #force = self.force_proj(
        #    x[:, 1:], attn_bias[:, :, 1:, 1:].reshape(-1, n_atoms, extend_n_atoms), delta_pos, outcell_index
        #)

        pos = batched_data['pos']
        n_graph, n_node = pos.size()[:2]
        padding_mask = ~batched_data["lig_mask"]
        is_not_pad = ~(padding_mask.reshape(-1))
        pos = pos.reshape(-1, 3)[is_not_pad]
        cnt = (~padding_mask).sum(-1)
        cnt_cumsum = cnt.cumsum(0)
        batch = torch.zeros(pos.shape[0]).to(cnt)
        batch[cnt_cumsum[:-1]] = 1
        batch = batch.cumsum(0)
        edge_index, edge_weight, edge_vec = self.distance(pos.to(torch.float32), batch)
        assert (
                edge_vec is not None
        ), "Distance module did not return directional information"
        edge_attr = self.distance_expansion(edge_weight)
        edge_mask = edge_index[0] != edge_index[1]
        edge_vec[edge_mask] = edge_vec[edge_mask] / (torch.norm(edge_vec[edge_mask], dim=1).unsqueeze(1) + 1e-3)
        x_feat = x.contiguous()[:, 1:].reshape(-1, self.args.encoder_embed_dim)[is_not_pad]
        vec_feat = torch.zeros(x_feat.size(0), 3, x_feat.size(1)).to(x_feat)
        edge_weight, edge_vec, edge_attr = \
            edge_weight.to(x_feat), edge_vec.to(x_feat), edge_attr.to(x_feat)
        for attn in self.attention_layers:
            dx, dvec = attn(x_feat, vec_feat, edge_index, edge_weight, edge_attr, edge_vec)
            x_feat = x_feat + dx
            vec_feat = vec_feat + dvec
        x_feat = self.out_norm(x_feat)
        if self.out_norm_vec is not None:
            vec_feat = self.out_norm_vec(vec_feat)
        new_atom_output = self.output_model_noise(x_feat, vec_feat)
        force = torch.zeros(n_graph, n_node, 3).to(new_atom_output)
        total = 0
        for atom_idx in range(n_graph):
            cur_valid_atoms = int((~padding_mask[atom_idx]).sum())
            force[atom_idx, :cur_valid_atoms, :] = new_atom_output[total:total + cur_valid_atoms, :]
            total += cur_valid_atoms

        #radius_score = torch.mean(self.radius_proj(x[:, 1:]), dim=1) # B x C
        radius_score = torch.zeros([n_atoms, self.num_radius_classes])

        # to avoid memory leak
        self.encoder.graph_encoder._attn_bias_3d = None
        self.encoder.graph_encoder._edge_features = None
        self.encoder.graph_encoder._bias = None

        # trick, use force as delta_pos prediction for loading pretrained is2rs model
        return force, radius_score

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

    def get_conditioned_derivative(self, batched_data):
        pos = batched_data["pos"] # B x T x 3
        device = pos.device
        n_graphs, n_tokens = pos.size()[:2]
        lig_mask = batched_data["lig_mask"]
        pos_masked = pos * lig_mask.unsqueeze(-1)
        dist = torch.norm(pos_masked.unsqueeze(1) - pos_masked.unsqueeze(2), dim=-1, p=2) # B x T x T
        max_pairs = torch.argmax(dist.reshape(n_graphs, -1), dim=-1)
        max_pair_i, max_pair_j = max_pairs // n_tokens, max_pairs % n_tokens
        max_dist = dist[torch.arange(n_graphs, device=device), max_pair_i, max_pair_j]
        grad_factor = -torch.sign(max_dist - self.target_radius) / (max_dist + 1e-32)
        pos_i = pos[torch.arange(n_graphs, device=device), max_pair_i, :].unsqueeze(1).repeat([1, n_tokens, 1])
        pos_j = pos[torch.arange(n_graphs, device=device), max_pair_j, :].unsqueeze(1).repeat([1, n_tokens, 1])
        mask_i = torch.arange(n_tokens, device=device).unsqueeze(0).repeat([n_graphs, 1]) == max_pair_i.unsqueeze(-1)
        mask_j = torch.arange(n_tokens, device=device).unsqueeze(0).repeat([n_graphs, 1]) == max_pair_j.unsqueeze(-1)
        grad = grad_factor.unsqueeze(1).unsqueeze(2) * (mask_i.unsqueeze(-1) * (pos_i - pos_j) + mask_j.unsqueeze(-1) * (pos_j - pos_i))
        return grad

    def calc_radius(self, batched_data):
        lig_mask = batched_data["lig_mask"]
        pos = batched_data["pos"].clone()
        pos_masked = pos * lig_mask.unsqueeze(-1)
        dist = torch.norm(pos_masked.unsqueeze(1) - pos_masked.unsqueeze(2), dim=-1, p=2) # B x T x T
        max_dist = torch.amax(dist, dim=[1, 2])
        return max_dist

  
    def complete_cell(self, batched_data, t=4999):
        device = batched_data["pos"].device
        dtype = batched_data["pos"].dtype
        cell_matrix = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=dtype, device=device)
        n_graphs, n_tokens = batched_data["pos"].size()[:2]
        gather_index = torch.tensor([0, 4, 2, 1], device=device, dtype=torch.long).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) + batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
        lattice = torch.gather(batched_data["pos"], 1, index=gather_index)
        corner = lattice[:, 0, :]
        lattice = lattice[:, 1:, :] - corner.unsqueeze(1)
        batched_data["cell_pred"] = lattice
        cell = torch.matmul(cell_matrix, lattice) + corner.unsqueeze(1)
        scatter_index = torch.arange(8, device=device).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) +\
            batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
        batched_data["pos"] = batched_data["pos"].scatter(1, scatter_index, cell)


    def complete_cell_for_pos(self, pos, batched_data):
        device = pos.device
        dtype = pos.dtype
        cell_matrix = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=dtype, device=device)
        n_graphs, n_tokens = pos.size()[:2]
        gather_index = torch.tensor([0, 4, 2, 1], device=device, dtype=torch.long).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) + batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
        lattice = torch.gather(pos, 1, index=gather_index)
        corner = lattice[:, 0, :]
        lattice = lattice[:, 1:, :] - corner.unsqueeze(1)
        cell = torch.matmul(cell_matrix, lattice) + corner.unsqueeze(1)
        scatter_index = torch.arange(8, device=device).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) +\
            batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
        pos = pos.scatter(1, scatter_index, cell)
        return pos


    def get_sampling_output(self, batched_data, pos_center=None, **kwargs):
        make_masks(batched_data)
        device = batched_data["x"].device

        n_graphs = batched_data["x"].shape[0]

        center_pos = get_center_pos(batched_data)
        batched_data["pos"] -= center_pos
        batched_data["pos"] = batched_data["pos"].masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)
        orig_pos = batched_data["pos"].clone()

        init_cell_pos = torch.zeros_like(batched_data["pos"])
        init_cell_pos_input = torch.tensor([[[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 1.0, 1.0],
                                            [1.0, 0.0, 0.0],
                                            [1.0, 0.0, 1.0],
                                            [1.0, 1.0, 0.0],
                                            [1.0, 1.0, 1.0]]],
                                           dtype=batched_data["pos"].dtype, device=batched_data["pos"].device).repeat([n_graphs, 1, 1]) * self.lattice_size -\
                                              (self.lattice_size / 2.0) # centering
        scatter_index = torch.arange(8, device=batched_data["pos"].device).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) +\
            batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
        init_cell_pos = init_cell_pos.scatter(1, scatter_index, init_cell_pos_input)
        batched_data["init_cell_pos"] = init_cell_pos.clone()

        # fill ligand pos with noise, keep protein pos
        pos_noise = torch.zeros(size=orig_pos.size(), device=device)
        if not self.no_diffusion:
            pos_noise = pos_noise.normal_() * self.diffusion_noise_std
  
        batched_data["pos"] = pos_noise + init_cell_pos
        batched_data["pos"] = batched_data["pos"].masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)
        self.complete_cell(batched_data, self.num_timesteps - 1)

        if not self.no_diffusion:
            if self.args.diffusion_sampling == "ddpm":
                # Sampling from Step T-1 to Step 0
                for t in tqdm(range(self.num_timesteps - 1, -1, -1)):
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
                    force, _ = self(batched_data, **kwargs)
                    force = force.detach()

                    epsilon = torch.zeros_like(batched_data["pos"]).normal_() * self.diffusion_noise_std

                    lig_pos = (
                        batched_data["pos"] - init_cell_pos - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * force
                    ) / alpha_t.sqrt() + sigma_t * epsilon

                    batched_data["pos"] = lig_pos + init_cell_pos
                    # update positions (only movable atons, marked by lig_mask)
                    self.complete_cell(batched_data, t)
                    batched_data["pos"] = batched_data["pos"].detach()
                    batched_data["pos"] = batched_data["pos"].masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)
                with open(f"pos_ddpm.pkl", "ab") as out_file:
                    pkl.dump((batched_data["sid"], batched_data["pos"] + center_pos), out_file)
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
                    force, _ = self.forward(batched_data, sampling=False, original_pos=None, **kwargs)
                    lig_pos = batched_data["pos"] - init_cell_pos
                    x_0_pred = (lig_pos - (1.0 - hat_alpha_t).sqrt() * force) / hat_alpha_t.sqrt()
                    epsilon = torch.zeros_like(lig_pos).normal_() * self.diffusion_noise_std
                    lig_pos = hat_alpha_t_1.sqrt() * x_0_pred +\
                        (1.0 - hat_alpha_t_1 - sigma_t ** 2).sqrt() * (lig_pos - hat_alpha_t.sqrt() * x_0_pred) / (1.0 - hat_alpha_t).sqrt() + sigma_t * epsilon
                    batched_data["pos"] = lig_pos + init_cell_pos # tensor_merge(batched_data["lig_mask"].unsqueeze(-1), lig_pos, batched_data["pos"])
                    self.complete_cell(batched_data)
                    batched_data["pos"] = batched_data["pos"].detach()
                    batched_data["pos"] = batched_data["pos"].masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)

                # forward for last step
                t = sampled_steps[0]
                hat_alpha_t = self.alphas[t]

                # forward
                batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long).fill_(t)
                force, _ = self.forward(batched_data, sampling=False, original_pos=None, **kwargs)
                lig_pos = batched_data["pos"] - init_cell_pos
                x_0_pred = (lig_pos - (1.0 - hat_alpha_t).sqrt() * force) / hat_alpha_t.sqrt()
                batched_data["pos"] = x_0_pred + init_cell_pos # tensor_merge(batched_data["lig_mask"].unsqueeze(-1), x_0_pred, batched_data["pos"])
                self.complete_cell(batched_data)
                batched_data["pos"] = batched_data["pos"].detach()
                batched_data["pos"] = batched_data["pos"].masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)
            elif self.diffusion_sampling == "ode":
                lattice_scatter_index = torch.tensor([[[4], [2], [1]]], device=device, dtype=torch.long).repeat([n_graphs, 1, 3])
                lattice_scatter_index += batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
                for t in tqdm(range(self.num_timesteps - 1, -1, -1)):
                    batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long).fill_(t)
                    beta_t = self.betas[t]
                    score, _ = self.forward(batched_data, sampling=False, original_pos=None, **kwargs)
                    score = -score / (1.0 - self.alphas[t]).sqrt()
                    score = score.masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)
                    pos = batched_data["pos"].clone() - init_cell_pos
                    epsilon = torch.zeros_like(pos).normal_()
                    if self.conditioned_ode_factor != 0.0:
                        bandgap, pos_derivatives, lattice_derivatives = get_bandgap_and_derivative_batched_multi_intervals(\
                            batched_data, t, center_pos, self.target_bandgap_interval, self.target_bandgap_softmax_temperature)
                        pos_derivatives = pos_derivatives.to(dtype=pos.dtype, device=pos.device)
                        lattice_derivatives = lattice_derivatives.to(dtype=pos.dtype, device=pos.device)
                        pos_derivatives = pos_derivatives.scatter(1, lattice_scatter_index, lattice_derivatives)
                    else:
                        pos_derivatives = 0.0
                    batched_data["pos"] = (2 - (1.0 - beta_t).sqrt()) * pos + 0.5 * beta_t * (score) - abs(self.conditioned_ode_factor) * pos_derivatives + init_cell_pos
                    self.complete_cell(batched_data, t)
                    batched_data["pos"] = batched_data["pos"].detach()
                    batched_data["pos"] = batched_data["pos"].masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)
                bandgap, pos_derivatives, lattice_derivatives = get_bandgap_and_derivative_batched_multi_intervals(\
                    batched_data, t, center_pos, self.target_bandgap_interval, self.target_bandgap_softmax_temperature)
                with open(f"{self.sampling_result_dir}/pos_ode_grad_clip_{self.conditioned_ode_factor}_int{self.target_bandgap_interval}_temp{self.target_bandgap_softmax_temperature}_gpu{self.gpu_device_id_record}_seed{self.seed_record}_no_lattice_grad.pkl", "ab") as out_file:
                    pkl.dump((bandgap, batched_data["sid"], batched_data["pos"] + center_pos, bandgap), out_file)
            else:
                raise ValueError(f"Unknown diffusion sampling strategy {self.args.diffusion_sampling}. Support only ddim and ddpm.")
        else:
            batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long)
            _, lig_pos = self.forward(batched_data, **kwargs)
            lig_pos = lig_pos.detach()
            batched_data["pos"] = lig_pos

        pred_pos = batched_data["pos"].clone()

        loss = ((pred_pos - orig_pos) ** 2).masked_fill(~batched_data["lig_mask_loss"].unsqueeze(-1), 0.0)
        loss = torch.sum(loss, dim=-1, keepdim=True)

        rmsd = torch.sqrt(torch.sum(loss, dim=-2) / (batched_data["natoms"][:, None] - 4))

        rmsd_pbc_val = rmsd_pbc(pred_pos, orig_pos, batched_data["cell"], batched_data["lig_mask_loss"], batched_data["natoms"] - 4)

        return {
            "pred_pos": pred_pos,
            "persample_loss": loss,
            "persample_rmsd": rmsd,
            "persample_rmsd_pbc": rmsd_pbc_val,
            "sample_size": n_graphs,
        }

    def get_training_output(self, batched_data, **kwargs):
        device = batched_data["x"].device
        n_graphs, _ = batched_data["x"].shape[:2]  # G, T
        # paddings:  (G, T)

        # postions centering (ligand center)

        pos = batched_data["pos"]
        center_pos = get_center_pos(batched_data)
        pos -= center_pos
        pos = pos.masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)

        init_cell_pos = torch.zeros_like(batched_data["pos"])
        init_cell_pos_input = torch.tensor([[[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 1.0, 1.0],
                                            [1.0, 0.0, 0.0],
                                            [1.0, 0.0, 1.0],
                                            [1.0, 1.0, 0.0],
                                            [1.0, 1.0, 1.0]]],
                                           dtype=batched_data["pos"].dtype, device=batched_data["pos"].device).repeat([n_graphs, 1, 1]) * self.lattice_size \
                                              - (self.lattice_size / 2.0) # centering
        scatter_index = torch.arange(8, device=batched_data["pos"].device).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) + \
                        batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
        init_cell_pos = init_cell_pos.scatter(1, scatter_index, init_cell_pos_input)
        batched_data["init_cell_pos"] = init_cell_pos.clone()


        orig_pos = pos.clone()

        if not self.no_diffusion:
            time_step = sample_time_step(n_graphs, self.num_timesteps, device)
            a = self.alphas.index_select(0, time_step)  # (G, )

            # Perturb pos, only on ligand atoms
            a_pos = a[:, None, None]  # (G, 1, 1)
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        if not self.no_diffusion:
            pos_noise = pos_noise.normal_() * self.diffusion_noise_std

        if not self.no_diffusion:
            a_pos_scale = a_pos.sqrt()
            pos_perturbed = (pos - init_cell_pos) * a_pos_scale + pos_noise * (1.0 - a_pos).sqrt() + init_cell_pos
        else:
            pos_perturbed = pos_noise

        batched_data["pos"] = pos_perturbed
        batched_data["pos"] = batched_data["pos"].masked_fill(~batched_data["lig_mask"].unsqueeze(-1), 0.0)
        self.complete_cell(batched_data)

        if not self.no_diffusion:
            batched_data["ts"] = time_step
        else:
            batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long)

        # forward
        pred_eps, radius_score = self(batched_data, **kwargs)

        if not self.no_diffusion:
            target_eps = pos_noise
        else:
            target_eps = orig_pos

        # calculate loss
        diff_loss = (pred_eps - target_eps) ** 2
        diff_loss = diff_loss.masked_fill(~batched_data["lig_mask_loss"].unsqueeze(-1), 0.0) # * (1.0 / a_pos_scale + 1.0)
        diff_loss = torch.sum(diff_loss, dim=-1, keepdim=True)

        if not self.no_diffusion:
            pred_pos = (batched_data["pos"] - init_cell_pos - pred_eps * (1.0 - a_pos).sqrt()) / a_pos_scale + init_cell_pos
            pred_pos = self.complete_cell_for_pos(pred_pos, batched_data)
        else:
            pred_pos = batched_data["pos"]

        pred_pos_rmsd = torch.sqrt(torch.sum(torch.sum((pred_pos - orig_pos) ** 2, dim=-1).masked_fill(~batched_data["lig_mask_loss"], 0.0), dim=-1) / (batched_data["natoms"] - 4))
        gather_index = torch.tensor([0, 4, 2, 1], device=device, dtype=torch.long).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) + batched_data["lnodes"].unsqueeze(-1).unsqueeze(-1)
        orig_pos_lattice = torch.gather(orig_pos, 1, index=gather_index)
        pred_pos_lattice = torch.gather(pred_pos, 1, index=gather_index)
        lattice_rmsd = torch.sqrt(torch.sum(torch.sum((orig_pos_lattice - pred_pos_lattice) ** 2, dim=-1), dim=-1) / 4.0)
        radius_loss = torch.zeros_like(diff_loss)

        return {
            "persample_loss": diff_loss,
            "persample_pred_pos_rmsd": pred_pos_rmsd,
            "persample_lattice_rmsd": lattice_rmsd,
            "persample_radius_loss": radius_loss,
            "sample_size": n_graphs,
        }


    def get_flow_ode_output(self, batched_data, **kwargs):
        make_masks(batched_data)
        n_graphs = batched_data["x"].size()[0]
        device = batched_data["x"].device
        lig_mask = batched_data["lig_mask"]
        init_center = get_init_center_pos(batched_data)
        batched_data["pos"] -= init_center
        batched_data["init_pos"] -= init_center
        batched_data["pred_pos"] -= init_center

        max_num_moveable = torch.max(batched_data["num_moveable"])
        index = torch.arange(n_graphs, device=device)
        trace_index = torch.arange(3 * max_num_moveable, device=device)

        def score_fn(x_pos, batched_data, t, max_num_moveable):
            pos_clone = batched_data["pos"]
            batched_data["pos"] = tensor_merge_truncated(lig_mask.unsqueeze(-1), x_pos, pos_clone, max_num_moveable)
            a = self.alphas[t]
            ret, _ = self(batched_data, **kwargs)
            ret = -1.0 * ret / (1.0 - a).sqrt()
            batched_data["pos"] = pos_clone
            return ret[:, :max_num_moveable, :]

        def ode_fn(t, pos_and_p_numpy):
            with torch.set_grad_enabled(True) and torch.autograd.set_detect_anomaly(True):
                device = batched_data["pos"].device
                round_t = np.int64(min(max(0, np.round(t * self.num_timesteps)), self.num_timesteps - 1))
                beta = self.betas[round_t]
                alpha_t = self.alphas[round_t]
                alpha_t_1 = 1.0 if round_t == 0 else self.alphas[round_t - 1]
                sigma = ((1.0 - alpha_t_1) / (1.0 - alpha_t) * beta).sqrt()
                pos_and_p = torch.tensor(pos_and_p_numpy, device=device)
                pos = torch.nn.parameter.Parameter(pos_and_p[:-n_graphs].reshape(n_graphs, max_num_moveable, 3).clone(), requires_grad=True)

                if self.num_epsilon_estimator <= 0:
                    reverse_flow_ode_fn = lambda pos: (-0.5 * (sigma ** 2) * score_fn(pos, batched_data, round_t, max_num_moveable) - 0.5 * beta * pos).\
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
                    for _ in range(self.num_epsilon_estimator):
                        epsilon = torch.zeros_like(batched_data["pos"][:, :max_num_moveable, :]).normal_()
                        pos = torch.nn.parameter.Parameter(pos_and_p[:-n_graphs].reshape(n_graphs, max_num_moveable, 3).clone(), requires_grad=True)
                        for_estimator = -0.5 * (sigma ** 2) * score_fn(pos, batched_data, round_t, max_num_moveable)
                        reverse_flow_ode = for_estimator - 0.5 * beta * pos
                        for_estimator = for_estimator.masked_fill((~lig_mask[:, :max_num_moveable]).unsqueeze(-1), 0.0)
                        reverse_flow_ode = reverse_flow_ode.masked_fill((~lig_mask[:, :max_num_moveable]).unsqueeze(-1), 0.0)
                        drift_prod_sum = torch.sum(torch.sum(epsilon * for_estimator, dim=[1, 2]))
                        grad = torch.autograd.grad(drift_prod_sum, pos)[0]
                        grad = grad.masked_fill((~lig_mask[:, :max_num_moveable]).unsqueeze(-1), 0.0)
                        estimated_trace = torch.sum(grad * epsilon, dim=[1, 2]) - 0.5 * beta * torch.sum(lig_mask[:, :max_num_moveable] * 3.0, dim=1)
                        estimated_traces.append(estimated_trace.unsqueeze(0))
                    estimated_traces = torch.cat(estimated_traces, dim=0)
                    trace = torch.mean(estimated_traces, dim=0)
            ret = torch.cat([reverse_flow_ode.reshape(-1), trace], dim=0).detach().cpu()
            return ret

        init = torch.cat([batched_data["pred_pos"][:, :max_num_moveable, :].reshape(-1), torch.zeros(n_graphs, dtype=torch.float64, device=device)], dim=0).cpu()
        solution = integrate.solve_ivp(ode_fn, (1e-5, 1.0), init, rtol=1e-5, atol=1e-5, method='RK45')
        sol = solution.y[:, -1]
        likelihood = sol[-n_graphs:]
        latent_pos = sol[:-n_graphs].reshape(n_graphs, max_num_moveable, 3) - batched_data["init_pos"][:, :max_num_moveable, :].cpu().numpy()
        lig_mask = lig_mask.unsqueeze(-1).cpu().numpy()
        latent_pos = latent_pos * lig_mask[:, :max_num_moveable, :]
        prior_log_p = -0.5 * np.sum(lig_mask * 3.0, axis=(1, 2)) * np.log(2 * np.pi) - np.sum((latent_pos ** 2) / 2.0, axis=(1, 2))
        likelihood += prior_log_p

        batched_data["pos"] += init_center
        batched_data["init_pos"] += init_center
        batched_data["pred_pos"] += init_center

        return {
            "persample_likelihood": torch.tensor(likelihood).to(device=device),
            "persample_prior_log_p": torch.tensor(prior_log_p).to(device=device),
            "latent_pos": torch.tensor(latent_pos).to(device=device),
            "sample_size": n_graphs,
        }


    def upgrade_state_dict(self, state_dict):
        named_parameters = {
            k: v
            for k, v in self.named_parameters()
        }

        for key in named_parameters:
            if key.find("radius") != -1 or key.find("sid_encoder") != -1:
                state_dict[key] = named_parameters[key].data
                print("Copying", key, "(from init value)")

        if self.remove_head:
            for key in named_parameters:
                if key.find("dist_feature_extractor_init") != -1:
                    copy_key = key.replace("dist_feature_extractor_init", "dist_feature_extractor")
                    state_dict[key] = state_dict[copy_key]
                    print("Copying", key, f"(from {copy_key} of pretrained model)")
            for key in named_parameters:
                if key.find("dist_feature_extractor") != -1 and key.find("dist_feature_extractor_init") == -1:
                    state_dict[key] = named_parameters[key].data
                    print("Copying", key, "(from init value)")
            for key in named_parameters:
                if key.find("force_proj_delta_pos") != -1:
                    state_dict[key] = named_parameters[key].data
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
