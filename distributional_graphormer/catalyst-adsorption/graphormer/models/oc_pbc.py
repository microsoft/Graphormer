import torch
from torch import nn
from ..modules.graphormer_3d_layer import NonLinear, GaussianLayer, RBF

class Graph3DBiasPBCCutoff(nn.Module):
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
        num_diffusion_timesteps,
        no_share_rpe=False,
    ):
        super(Graph3DBiasPBCCutoff, self).__init__()
        self.num_heads = num_heads
        self.num_atom_types = num_atom_types
        self.num_layers = num_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.num_diffusion_timesteps = num_diffusion_timesteps
        num_edge_types = num_atom_types * num_atom_types

        rpe_heads = (
            self.num_heads * self.num_layers if self.no_share_rpe else self.num_heads
        )
        if dist_feature_extractor == "gbf" and dist_feature_extractor == "rbf":
            raise ValueError("dist_feature_extractor can only be gbf or rbf")
        self.dist_feature_extractor = GaussianLayer(50, self.num_kernel, num_edge_types) if dist_feature_extractor == "gbf" \
                                      else RBF(50, self.num_kernel, num_edge_types)
        self.dist_feature_extractor_init = GaussianLayer(50, self.num_kernel, num_edge_types) if dist_feature_extractor == "gbf" \
                                      else RBF(50, self.num_kernel, num_edge_types)
        self.feature_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

    def get_attn_bias(self, pos, padding_mask, expand_pos, expand_mask, n_node, expand_n_node, edge_types, t, is_init):
        pos = pos.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        expand_pos = expand_pos.masked_fill(
            expand_mask.unsqueeze(-1).to(torch.bool), 0.0
        )
        expand_pos = torch.cat([pos, expand_pos], dim=1)    

        delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1) # B x T x (expand T) x 3
        dist = delta_pos.norm(dim=-1).view(-1, n_node, expand_n_node)
        full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
        #ic(full_mask.size(), dist.size(), padding_mask.size(), delta_pos.size())
        dist = dist.masked_fill(
            full_mask.unsqueeze(1).to(torch.bool), 1.0
        )
        dist = dist.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 1.0
        )
        delta_pos_norm = delta_pos / (dist.unsqueeze(-1) + 1e-5)

        if is_init:
            edge_feature = self.dist_feature_extractor_init(dist, edge_types, t)
        else:
            edge_feature = self.dist_feature_extractor(dist, edge_types, t)

        graph_attn_bias = self.feature_proj(edge_feature)

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()

        padding_mask = torch.cat([padding_mask, expand_mask], dim=1)
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)
        return dist, delta_pos_norm, graph_attn_bias, merge_edge_features


    def forward(self, batched_data):
        init_pos, pos, x, t, expand_pos, init_expand_pos, expand_mask, outcell_index = (
            batched_data["init_pos"],
            batched_data["pos"],
            batched_data["x"],
            batched_data["ts"],
            batched_data["expand_pos"],
            batched_data["init_expand_pos"],
            batched_data["expand_mask"],
            batched_data["outcell_index"],
        )  # pos shape: [n_graphs, n_nodes, 3]
        n_graph, n_node, _ = pos.shape

        atoms = x[:, :, 0]
        expand_atoms = torch.gather(atoms, dim=1, index=outcell_index)
        expand_atoms = torch.cat([atoms, expand_atoms], dim=1)
        expand_n_node = expand_atoms.size()[1]
        edge_types = atoms.view(n_graph, n_node, 1) * self.num_atom_types + expand_atoms.view(
            n_graph, 1, expand_n_node
        )
        padding_mask = atoms.eq(0)  # (G, T)

        dist, delta_pos, graph_attn_bias, merge_edge_features = self.get_attn_bias(
            pos, padding_mask, expand_pos, expand_mask, n_node, expand_n_node, edge_types, t, False)
        _, _, init_graph_attn_bias, init_merge_edge_features = self.get_attn_bias(
            init_pos, padding_mask, init_expand_pos, expand_mask, n_node, expand_n_node, edge_types, t, True)

        return {
            "dist": dist,
            "delta_pos": delta_pos,
            "attn_bias_3d": graph_attn_bias + init_graph_attn_bias,
            "edge_features": merge_edge_features + init_merge_edge_features,
        }
