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

import os
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
from ..modules import Graph3DBias, NodeTaskHead
from .graphormer_encoder import GraphormerEncoder as GraphormerEncoderBase

from .model_utils import (
    mask_after_k_persample,
    make_masks,
    get_center_pos,
    tensor_merge,
)

from .diffusion.schedulers.legacy_scheduler import get_beta_schedule

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


class TimeStepEncoder(nn.Module):
    def __init__(self, args):
        super(TimeStepEncoder, self).__init__()
        self.args = args

        if args.diffusion_timestep_emb_type == "positional":
            self.time_proj = SinusoidalPositionEmbeddings(args.encoder_embed_dim)
        else:
            raise NotImplementedError

        self.time_embedding = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
            nn.GELU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )

    def forward(self, timesteps):
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        return t_emb


class GraphormerGraphEncoder(GraphormerGraphEncoderBase):
    """
    User-defined extra node layers or bias layers
    """

    def init_extra_node_layers(self, args):
        super().init_extra_node_layers(args)
        self.timestep_encoder = TimeStepEncoder(args)
        self.dist_feature_node_scale = args.dist_feature_node_scale
        self.tag_embedding = nn.Embedding(3, args.encoder_embed_dim)
        self.no_diffusion = args.no_diffusion

    def init_extra_bias_layers(self, args):
        super().init_extra_bias_layers(args)

    def forward_extra_node_layers(self, batched_data, x):
        x = super().forward_extra_node_layers(batched_data, x)
        assert x.shape[0] == batched_data["ts"].shape[0]

        if not self.no_diffusion:
            # get time embedding
            ts = batched_data["ts"].to(x.device)  # [B, ]
            time_emb = self.timestep_encoder(ts).type_as(x)  # [B, d]
            batched_data["time_emb"] = time_emb

            # add time embedding
            x += time_emb[:, None, :]
        else:
            batched_data["time_emb"] = None

        # add edge feature
        x[:, 1:, :] += batched_data["_edge_features"] * self.dist_feature_node_scale / 2
        x[:, 1:, :] += batched_data["_edge_features_crystal"] * self.dist_feature_node_scale / 2 # need to mask ligand part

        # add tag embedding
        atoms = batched_data["x"][:, :, 0]
        padding_mask = atoms.eq(0)
        n_graph, n_node = atoms.size()[:2]
        lnode = batched_data["lnode"]
        tag_mask = ~mask_after_k_persample(n_graph, n_node, lnode)
        tag_mask = tag_mask.masked_fill(padding_mask, 2).long()
        tag_features = self.tag_embedding(tag_mask)
        x[:, 1:, :] += tag_features

        batched_data["_edge_features"] = None
        return x

    def forward_extra_bias_layers(self, batched_data, attn_bias):
        bias = super().forward_extra_bias_layers(batched_data, attn_bias)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + batched_data["_attn_bias_3d"]

        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + batched_data["_attn_bias_3d_crystal"] # need to mask ligand part
        bias /= 3
        batched_data["_attn_bias_3d_crystal"] = None
        batched_data["_bias"] = bias
        batched_data["_attn_bias_3d"] = None
        return bias


class GraphormerEncoder(GraphormerEncoderBase):
    def build_graph_encoder(self, args):
        return GraphormerGraphEncoder(args)


@dataclass
class GraphormerDiffModelConfig(FairseqDataclass):
    num_diffusion_timesteps: int = field(
        default=5000, metadata={"help": "number of diffusion timesteps"}
    )

    diffusion_timestep_emb_type: str = field(
        default="positional",
        metadata={"help": "type of time embedding for diffusion timesteps"},
    )

    diffusion_layer_add_time_emb: bool = field(
        default=False,
        metadata={"help": "whether to add time embedding to node features layerwise"},
    )

    diffusion_layer_proj_time_emb: bool = field(
        default=False,
        metadata={"help": "whether to project time embedding layerwise"},
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
        default="ddpm",
        metadata={"help": "sampling strategy, ddpm or ddim"},
    )
    ddim_steps: int = field(
        default=50,
        metadata={"help": "sampling steps for ddim"},
    )
    ddim_eta: float = field(
        default=0.0,
        metadata={"help": "eta for ddim"},
    )
    num_atom_types: int = field(
        default=128,
        metadata={"help": "number of atom types"},
    )
    dist_feature_extractor: str = field(
        default="rbf",
        metadata={"help": "distance feature extractor, can be rbf or gbf"},
    )
    dist_feature_node_scale: float = field(
        default=1.0,
        metadata={"help": "scale of distance feature added to node representations"},
    )
    dist_feature_num_kernels: int = field(
        default=128,
        metadata={"help": "number of kernels to extract distance features"},
    )
    prior_distribution_std: float = field(
        default=1,
        metadata={"help": "variance of prior distribution"},
    )
    reweighting_file: str = field(
        default="",
        metadata={
            "help": "using reweighting file to reweight the loss according to RMSD_to_crystal"
        },
    )
    no_diffusion: bool = field(
        default=False, metadata={"help": "disable diffusion and training on bare graphormer"},
    )
    ligand_only: bool = field(
        default=False, metadata={"help": "using water-ligand dataset"},
    )
    protein_only: bool = field(
        default=False, metadata={"help": "using protein only dataset"},
    )
    ligand_center: bool = field(
        default=False, metadata={"help": "using ligand center instead of protein center"},
    )
    pairwise_loss: bool = field(
        default=False, metadata={"help": "reweighting loss using atom numbers per protein-ligand pair"},
    )
    test_mode: bool = field(
        default=False, metadata={"help": "switch to test mode to generate conformers"},
    )
    num_epsilon_estimator: int = field(
        default=8, metadata={"help": "number of epsilons to sampled for trace estimation in flow ode"},
    )



@register_model("graphormer_diff", dataclass=GraphormerDiffModelConfig)
class GraphormerDiffModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)

        self.init_diffusion(args)

        self.graph_3d_bias = Graph3DBias(
            num_heads=args.encoder_attention_heads,
            num_atom_types=args.num_atom_types,
            num_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            num_kernel=args.dist_feature_num_kernels,
            dist_feature_extractor=args.dist_feature_extractor,
            no_share_rpe=False,
        )

        self.graph_3d_bias_crystal = Graph3DBias(
            num_heads=args.encoder_attention_heads,
            num_atom_types=args.num_atom_types,
            num_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            num_kernel=args.dist_feature_num_kernels,
            dist_feature_extractor=args.dist_feature_extractor,
            no_share_rpe=False,
        )

        self.force_proj = NodeTaskHead(
            args.encoder_embed_dim, args.encoder_attention_heads
        )

        self.diffusion_sampling = self.args.diffusion_sampling
        self.ddim_steps = self.args.ddim_steps
        self.ddim_eta = self.args.ddim_eta
        self.prior_distribution_std = self.args.prior_distribution_std
        self.pairwise_loss = self.args.pairwise_loss
        self.reweighting_file = self.args.reweighting_file
        self.no_diffusion = self.args.no_diffusion
        self.ligand_only = self.args.ligand_only
        self.protein_only = self.args.protein_only
        self.ligand_center = self.args.ligand_center
        self.num_epsilon_estimator = self.args.num_epsilon_estimator

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
        n_atoms = batched_data["x"].shape[1]
        graph_3d = self.graph_3d_bias(batched_data)
        graph_3d_crystal = self.graph_3d_bias_crystal(batched_data, crystal=True)
        delta_pos = graph_3d["delta_pos"]

        # (G, H, T, T)
        batched_data["_attn_bias_3d"] = graph_3d["attn_bias_3d"]
        # (G, T, D)
        batched_data["_edge_features"] = graph_3d["edge_features"]
        batched_data["_attn_bias_3d_crystal"] = graph_3d_crystal["attn_bias_3d_crystal"]
        batched_data["_edge_features_crystal"] = graph_3d_crystal["edge_features_crystal"]

        encoder_out = self.encoder(batched_data, **kwargs)

        x = encoder_out["encoder_out"][0].transpose(0, 1)  # B x T x D
        attn_bias = batched_data["_bias"]
        force = self.force_proj(
            x[:, 1:], attn_bias[:, :, 1:, 1:].reshape(-1, n_atoms, n_atoms), delta_pos
        )

        # to avoid memory leak
        batched_data["_time_emb"] = None
        batched_data["_attn_bias_3d"] = None
        batched_data["_edge_features"] = None
        batched_data["_attn_bias_3d_crystal"] = None
        batched_data["_edge_features_crystal"] = None
        batched_data["_bias"] = None
        return force

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

    def get_sampling_output(
        self, batched_data, pos_center=None, sampling_times=1, **kwargs
    ):
        make_masks(batched_data)

        device = batched_data["x"].device

        n_graphs, _ = batched_data["x"].shape[:2]
        lig_mask = batched_data["lig_mask"]
        pro_mask = batched_data["pro_mask"]
        orig_pos = batched_data["pos"]

        # centering
        if pos_center is None:
            if self.ligand_only or self.ligand_center:
                pos_center = get_center_pos(batched_data,'ligand')
            else:
                pos_center = get_center_pos(batched_data)

        orig_pos -= pos_center

        pred_pos_list = []
        for st in range(sampling_times):
            # fill ligand pos with noise, keep protein pos
            if self.no_diffusion:
                pos_noise = torch.zeros(size=orig_pos.size(), device=device)
            else:
                pos_noise = torch.zeros(size=orig_pos.size(), device=device).normal_(0,self.prior_distribution_std)

            pos = pos_noise

            # protein pos and noisy ligand pos
            batched_data["pos"] = pos

            crystal_pos_center = get_center_pos(batched_data,crystal=True)
            batched_data["crystal_pos"] -= crystal_pos_center
            batched_data_crystal_pos_saver = batched_data["crystal_pos"]
            # replace batched_data["crystal_pos"] lnode part with batched_data["pos"] lnode part
            batched_data["crystal_pos"] = tensor_merge(lig_mask[:, :, None], batched_data["pos"], batched_data["crystal_pos"])

            if not self.no_diffusion:
                if self.diffusion_sampling == "ddpm":
                    # Sampling from Step T-1 to Step 0
                    for t in range(self.num_timesteps - 1, -1, -1):
                        hat_alpha_t = self.alphas[t]
                        hat_alpha_t_1 = 1.0 if t == 0 else self.alphas[t - 1]
                        alpha_t = hat_alpha_t / hat_alpha_t_1
                        beta_t = 1 - alpha_t
                        sigma_t = (
                            0.0
                            if t == 0
                            else (
                                (1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t
                            ).sqrt()
                        )

                        # forward
                        batched_data["ts"] = torch.ones(n_graphs, device=device).fill_(t)
                        force = self(batched_data, **kwargs).detach()

                        epsilon = torch.zeros_like(batched_data["pos"]).normal_(0,self.prior_distribution_std)

                        lig_pos = (
                            batched_data["pos"]
                            - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * force * self.prior_distribution_std
                        ) / alpha_t.sqrt() + sigma_t * epsilon


                        batched_data["pos"] = lig_pos
                        batched_data["crystal_pos"] = tensor_merge(lig_mask[:, :, None], batched_data["pos"], batched_data["crystal_pos"])
                        batched_data["pos"] = batched_data["pos"].detach()
                elif self.diffusion_sampling == "ddim":
                    sampled_steps, _ = torch.sort(
                        (
                            torch.randperm(
                                self.num_timesteps - 2, dtype=torch.long, device=device
                            )
                            + 1
                        )[: self.ddim_steps - 1]
                    )
                    sampled_steps = torch.cat(
                        [
                            sampled_steps,
                            torch.tensor([self.num_timesteps - 1], device=device).long(),
                        ]
                    )
                    for i in range(sampled_steps.shape[0] - 1, 0, -1):
                        t = sampled_steps[i]
                        t_1 = sampled_steps[i - 1]
                        hat_alpha_t = self.alphas[t]
                        hat_alpha_t_1 = self.alphas[t_1]
                        alpha_t = hat_alpha_t / hat_alpha_t_1
                        beta_t = 1.0 - alpha_t
                        sigma_t = (
                            self.ddim_eta
                            * ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt()
                        )

                        # forward
                        batched_data["ts"] = torch.zeros(n_graphs, device=device).fill_(t)
                        force = self(batched_data, **kwargs)
                        lig_pos = batched_data["pos"]
                        x_0_pred = (
                            lig_pos - (1.0 - hat_alpha_t).sqrt() * force * self.prior_distribution_std
                        ) / hat_alpha_t.sqrt()
                        epsilon = torch.zeros_like(lig_pos).normal_(0,self.prior_distribution_std)
                        lig_pos = (
                            hat_alpha_t_1.sqrt() * x_0_pred
                            + (1.0 - hat_alpha_t_1 - sigma_t**2).sqrt()
                            * (lig_pos - hat_alpha_t.sqrt() * x_0_pred)
                            / (1.0 - hat_alpha_t).sqrt()
                            + sigma_t * epsilon
                        )

                        batched_data["pos"] = lig_pos
                        batched_data["crystal_pos"] = tensor_merge(lig_mask[:, :, None], batched_data["pos"], batched_data["crystal_pos"])
                        batched_data["pos"] = batched_data["pos"].detach()

                    # forward for last step
                    t = sampled_steps[0]
                    hat_alpha_t = self.alphas[t]

                    # forward
                    batched_data["ts"] = torch.zeros(n_graphs, device=device).fill_(t)
                    force = self(batched_data, **kwargs)
                    lig_pos = batched_data["pos"]
                    x_0_pred = (
                        lig_pos - (1.0 - hat_alpha_t).sqrt() * force * self.prior_distribution_std
                    ) / hat_alpha_t.sqrt()

                    batched_data["pos"] = x_0_pred
                    batched_data["pos"] = batched_data["pos"].detach()
                else:
                    raise ValueError(
                        f"Unknown sampling strategy {self.args.sampling}. Support only ddim and ddpm."
                    )
            else:
                batched_data["ts"] = torch.zeros(n_graphs, device=device, dtype=torch.long)
                lig_pos = self.forward(batched_data, **kwargs).detach()
                batched_data["pos"] = tensor_merge(lig_mask[:, :, None], lig_pos, batched_data["pos"])

            pred_pos = batched_data["pos"]

            pred_pos_list.append(pred_pos)

        pred_pos = torch.stack(pred_pos_list, dim=0).mean(dim=0)


        if self.ligand_only or self.protein_only:
            pred_pos, orig_pos = self.rigid_transform_Kabsch_3D_torch4batch(pred_pos, \
                orig_pos.float(), lig_mask, pro_mask)

        loss = (pred_pos - orig_pos) ** 2
        loss = torch.sum(loss, dim=-1, keepdim=True)

        loss_crystal = (pred_pos - batched_data_crystal_pos_saver) ** 2
        loss_crystal = torch.sum(loss_crystal, dim=-1, keepdim=True)
        loss_lig = loss_crystal.masked_fill((~lig_mask)[:, :, None], 0.0)
        rmsd_lig = torch.sqrt(torch.sum(loss_lig, dim=-2) / batched_data["lnode"][:, None])
        loss_pro = loss_crystal.masked_fill((~pro_mask)[:, :, None], 0.0)
        rmsd_pro = torch.sqrt(torch.sum(loss_pro, dim=-2) / batched_data["pnode"][:, None])
        loss = loss.masked_fill((~(pro_mask | lig_mask))[:, :, None], 0.0)
        rmsd = torch.sqrt(torch.sum(loss, dim=-2) / (batched_data["lnode"]+batched_data["pnode"])[:, None])

        if self.args.test_mode:
            if not hasattr(self, "fileid"):
                self.fileid = 0
            if not os.path.exists('./position_pt'):
                os.mkdir('./position_pt')
            #save torch tensor
            if self.ligand_only:
                torch.save(pred_pos+pos_center, './position_pt/pred_pos'+'_'+str(self.fileid)+'.pt')
                torch.save(orig_pos+pos_center, './position_pt/orig_pos'+'_'+str(self.fileid)+'.pt')
            else:
                torch.save(pred_pos+pos_center, './position_pt/pred_pos'+'_'+str(self.fileid)+'.pt')
                torch.save(orig_pos+pos_center, './position_pt/orig_pos'+'_'+str(self.fileid)+'.pt')
                torch.save(batched_data["lnode"], './position_pt/lnode'+'_'+str(self.fileid)+'.pt')
                torch.save(batched_data["pnode"], './position_pt/pnode'+'_'+str(self.fileid)+'.pt')
                torch.save(rmsd, './position_pt/rmsd'+'_'+str(self.fileid)+'.pt')
            self.fileid += 1

        return {
            "orig_pos": orig_pos+pos_center,
            "pred_pos": pred_pos+pos_center,
            "persample_loss": loss,
            "persample_rmsd_lig": rmsd_lig,
            "persample_rmsd_pro": rmsd_pro,
            "persample_rmsd": rmsd,
            "sample_size": n_graphs,
        }

    def rigid_transform_Kabsch_3D_torch4batch(self, pred, orig, lig_mask, pro_mask):

        def rigid_transform_Kabsch_3D_torch(A, B):
            # R = 3x3 rotation matrix, t = 3x1 column vector
            # This already takes residue identity into account.
            assert A.shape[1] == B.shape[1]
            num_rows, num_cols = A.shape
            if num_rows != 3:
                raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
            num_rows, num_cols = B.shape
            if num_rows != 3:
                raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
            # find mean column wise: 3 x 1
            centroid_A = torch.mean(A, axis=1, keepdims=True)
            centroid_B = torch.mean(B, axis=1, keepdims=True)
            # subtract mean
            Am = A - centroid_A
            Bm = B - centroid_B
            H = Am @ Bm.T
            # find rotation
            # U, S, Vt = torch.linalg.svd(H)
            # change above svd into numpy inplementation
            U, S, Vt = np.linalg.svd(H.cpu().numpy())
            U = torch.from_numpy(U).to(A.device)
            S = torch.from_numpy(S).to(A.device)
            Vt = torch.from_numpy(Vt).to(A.device)
            R = Vt.T @ U.T
            # special reflection case
            if torch.linalg.det(R) < 0:
                # print("det(R) < R, reflection detected!, correcting for it ...")
                SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
                R = (Vt.T @ SS) @ U.T
            assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher
            t = -R @ centroid_A + centroid_B
            return R, t

        batch_size = pred.shape[0]
        for i in range(batch_size):
            valid_atoms = pro_mask[i] | lig_mask[i]
            pred_i = pred[i][valid_atoms]
            orig_i = orig[i][valid_atoms]
            try:
                R, t = rigid_transform_Kabsch_3D_torch(pred_i.T, orig_i.T)
            except:
                print("Rigid transform failed, skip")
                continue
            pred_i = pred_i @ R.T + t.T

            '''
            There is a bug in model that we'll get mirrored sample results
            Here we try to fix it by comparing the rmsd between predicted results and mirrored predicted results
            Not very neat, but it works
            '''
            R_m, t_m = rigid_transform_Kabsch_3D_torch(-pred_i.T, orig_i.T)
            pred_mi = -pred_i @ R_m.T + t_m.T
            # calc rmsd between pred_i and orig_i
            rmsd1 = torch.sqrt(torch.sum((pred_i - orig_i) ** 2, dim=1).mean())
            rmsd2 = torch.sqrt(torch.sum((pred_mi - orig_i) ** 2, dim=1).mean())
            # copy pred_i back to pred
            pred[i][valid_atoms] = pred_i if rmsd1 < rmsd2 else pred_mi
        return pred,orig



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
    args.num_epsilon_estimator = safe_getattr(args, "num_epsilon_estimator", 10)


@register_model_architecture("graphormer_diff", "graphormer_diff_xs")
def small_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 64)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 64)
    base_architecture(args)
