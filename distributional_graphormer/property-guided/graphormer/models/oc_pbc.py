import torch
from torch_scatter import segment_coo, segment_csr
import numpy as np
from torch import nn, Tensor
from ..modules.graphormer_3d_layer import gaussian, NonLinear, GaussianLayer, RBF
from typing import Callable
from torch_scatter import scatter_sum

class GaussianLayerFlatten(nn.Module):
    def __init__(self, num_diffusion_timesteps, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(num_diffusion_timesteps, K)
        self.stds = nn.Embedding(num_diffusion_timesteps, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types, t):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, self.K)
        mean = self.means(t)
        std = self.stds(t).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class RBFFlatten(nn.Module):
    def __init__(self, num_diffusion_timesteps, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(num_diffusion_timesteps, K)
        self.temps = nn.Embedding(num_diffusion_timesteps, K)
        self.mul: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        self.bias: Callable[..., Tensor] = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.temps.weight, 0.1, 10)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: Tensor, edge_types, t):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        mean = self.means(torch.zeros_like(t))
        temp = self.temps(torch.zeros_like(t)).abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means.weight)


def radius_graph_pbc(
    data, radius, max_num_neighbors_threshold, max_pbc_cell_rep_threshold, pbc=[True, True, False]
):
    device = data["pos"].device
    batch_size = len(data["natoms"])
    bs, _, pos_dim = data["pos"].size()
    assert bs == batch_size and pos_dim == 3 

    # position of the atoms
    atom_pos = torch.cat([pos[:natom] for pos, natom in zip(data["pos"], data["natoms"])], axis=0)

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data["natoms"]
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data["cell"][:, 1], data["cell"][:, 2], dim=-1)
    cell_vol = torch.sum(data["cell"][:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data["cell"].new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data["cell"][:, 2], data["cell"][:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data["cell"].new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data["cell"][:, 0], data["cell"][:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data["cell"].new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [min(rep_a1.max(), max_pbc_cell_rep_threshold),
               min(rep_a2.max(), max_pbc_cell_rep_threshold),
               min(rep_a3.max(), max_pbc_cell_rep_threshold)]


    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data["cell"], 1, 2)
    if data_cell.dtype == torch.float16:
        pbc_offsets = torch.bmm(data_cell, unit_cell_batch.half())
    else:
        pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data["natoms"],
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image, max_rep


def get_max_neighbors_mask(
    natoms, index, atom_distance, max_num_neighbors_threshold
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    if max_num_neighbors_threshold >= 0:
        num_neighbors_thresholded = num_neighbors.clamp(
            max=max_num_neighbors_threshold
        )
    else:
        num_neighbors_thresholded = num_neighbors.clone()

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets=False,
    return_distance_vec=False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances))#[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]
    cell_offsets = cell_offsets[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
        "cell_offsets": cell_offsets,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def gen_edges_from_batch(batched_data, radius, max_num_neighbors_threshold, max_pbc_cell_rep_threshold):
    edge_index, cell_offsets, neighbors, max_rep = radius_graph_pbc(batched_data, radius, max_num_neighbors_threshold, max_pbc_cell_rep_threshold)
    out = get_pbc_distances(
        batched_data["pos"].view(-1, 3),
        edge_index,
        batched_data["cell"],
        cell_offsets,
        neighbors,
        return_offsets=True,
        return_distance_vec=True,
    )
    edge_index = out["edge_index"]
    edge_dist = out["distances"]
    cell_offsets = out["cell_offsets"]
    device = batched_data["pos"].device
    edge_cell_type = cell_offsets.clone()
    max_rep_tensor = torch.tensor(max_rep, device=device)

    edge_cell_type += max_rep_tensor
    num_rep_tensor = 2 * max_rep_tensor + 1
    max_rep_offsets = torch.tensor([num_rep_tensor[1] * num_rep_tensor[2], num_rep_tensor[1], 1], device=device)
    edge_cell_type = torch.sum(edge_cell_type * max_rep_offsets, axis=-1)
    return edge_cell_type, edge_dist, edge_index, neighbors


class Graph3DBiasPBC(nn.Module):
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
        max_pbc_cell_rep_threshold,
        pbc_radius,
        max_num_neighbors_threshold,
        no_share_rpe=False,
    ):
        super(Graph3DBiasPBC, self).__init__()
        self.num_heads = num_heads
        self.num_atom_types = num_atom_types
        self.num_layers = num_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.max_pbc_cell_rep_threshold = max_pbc_cell_rep_threshold
        self.num_pbc_cell_types = (2 * max_pbc_cell_rep_threshold + 1) ** 2
        self.pbc_radius = pbc_radius
        self.max_num_neighbors_threshold = max_num_neighbors_threshold
        num_edge_types = num_atom_types * num_atom_types * ((2 * max_pbc_cell_rep_threshold + 1) ** 2)

        rpe_heads = (
            self.num_heads * self.num_layers if self.no_share_rpe else self.num_heads
        )
        if dist_feature_extractor == "gbf" and dist_feature_extractor == "rbf":
            raise ValueError("dist_feature_extractor can only be gbf or rbf")
        self.dist_feature_extractor = GaussianLayerFlatten(self.num_diffusion_timesteps,
                                                    self.num_kernel, num_edge_types) if dist_feature_extractor == "gbf" \
                                      else RBFFlatten(self.num_diffusion_timesteps,
                                               self.num_kernel, num_edge_types)
        self.feature_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None


    def check_edge_index(self, edge_index, batch_size, max_num_atoms, scatter_edge_index):
        ones = torch.ones_like(edge_index[0, :], dtype=torch.long)
        num_edges = scatter_sum(ones, scatter_edge_index, dim=0, dim_size=batch_size * max_num_atoms * max_num_atoms).reshape(batch_size, max_num_atoms, max_num_atoms)
        # checks that edges are symmetric
        assert torch.sum((num_edges - torch.transpose(num_edges, 1, 2)) ** 2) == 0.0, f"{torch.sum((num_edges - torch.transpose(num_edges, 1, 2)) ** 2)}"

    def forward(self, batched_data):
        x, t, atomic_numbers, natoms, pos = (
            batched_data["x"],
            batched_data["ts"],
            batched_data["atomic_numbers"] + 1, # +1 to be consistent with x
            batched_data["natoms"],
            batched_data["pos"],
        )  # x shape: [n_graphs, n_nodes, _]
        device = x.device
        edge_cell_type, edge_dist, edge_index, neighbors = gen_edges_from_batch(batched_data, self.pbc_radius, self.max_num_neighbors_threshold, self.max_pbc_cell_rep_threshold)

        padding_mask = x[:, :, 0].eq(0)
        batch_size, num_max_atoms = x.size()[:2]

        atoms = atomic_numbers[edge_index[0, :]] * self.num_atom_types + atomic_numbers[edge_index[1, :]]
        edge_types = atoms * self.num_pbc_cell_types + edge_cell_type.long()

        edge_feature = self.dist_feature_extractor(edge_dist, edge_types, torch.repeat_interleave(t, neighbors))
        edge_index_offset = torch.cumsum(natoms, dim=0) - natoms
        edge_index_offset = torch.repeat_interleave(edge_index_offset, neighbors)
        edge_index_per_image = edge_index - edge_index_offset
        assert torch.sum(neighbors) == edge_index.size()[-1], f"torch.sum(neighbors) = {torch.sum(neighbors)}, edge_index.size() = {edge_index.size()}"
        scatter_edge_offset = torch.repeat_interleave(
            torch.arange(batch_size, device=device) * num_max_atoms * num_max_atoms, neighbors)
        scatter_edge_index = (edge_index_per_image[0, :] * num_max_atoms + edge_index_per_image[1, :]) + scatter_edge_offset
        edge_feature_gathered = scatter_sum(edge_feature, scatter_edge_index, dim=0, dim_size=batch_size * num_max_atoms * num_max_atoms)
        edge_feature_gathered = edge_feature_gathered.resize(batch_size, num_max_atoms, num_max_atoms, self.num_kernel)

        # self.check_edge_index(edge_index, batch_size, num_max_atoms, scatter_edge_index)

        graph_attn_bias = self.feature_proj(edge_feature_gathered)

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        edge_feature_gathered = edge_feature_gathered.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature_gathered.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, num_max_atoms, num_max_atoms)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        return {
            "attn_bias_3d": graph_attn_bias,
            "edge_features": merge_edge_features,
            "delta_pos": delta_pos,
        }


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
            batched_data["init_cell_pos"],
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
