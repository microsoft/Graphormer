from .diff_dataset import CrossDockedDM, CrossDockedDataset, CrossDockedRawDataset
from .diff_dataset import crossdocked_batch_collater
from .diff_dataset import EpochShuffleDataset

from ..collator import (
    pad_1d_unsqueeze,
    pad_2d_unsqueeze,
    pad_all_poses_unsqueeze,
    pad_pos_unsqueeze,
    pad_3d_unsqueeze,
    pad_attn_bias_unsqueeze,
    pad_edge_type_unsqueeze,
    pad_spatial_pos_unsqueeze,
)

import torch

class OCRawDataset(CrossDockedRawDataset):
    def __init__(self, path, reweighting_file, split, file_list, train_subset,  need_all_poses=False, subdir="full"):
        super().__init__(path, reweighting_file, split, file_list, train_subset, need_all_poses, subdir)
        # print(split)
        # self.split = split

class MDKDEEvalDatasetDM(CrossDockedDM):
    def get_split(self, split, train_subset):
        if split not in self.split_lut:
            split_list_lines = open(self.path + f"/{split}.list", "r").readlines()
            file_list = self.process_file_list(split_list_lines)
            self.split_lut[split] = OCRawDataset(self.path, self.reweighting_file, split, file_list, train_subset, self.need_all_poses)
        return self.split_lut[split]

class MDKDEEvalDataset(CrossDockedDataset):
    def collater(self, samples):
        if self.raw_dataset.need_all_poses:
            return md_batch_collater(samples, has_all_poses=True)
        else:
            return md_batch_collater(samples, has_all_poses=False)

def md_batch_collater(
    items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, has_all_poses=False
):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            torch.cat([pos.unsqueeze(0) for pos in item.all_poses], axis=0) if has_all_poses else None,
            item.pos,
            item.crystal_pos if 'crystal_pos' in items[0].keys else torch.rand_like(item.pos),
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.num_node,
            item.rmsd_to_crystal,
            item.pdbid,
        )
        for item in items
    ]
    (
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        all_poseses,
        poss,
        crystal_poss,
        edge_inputs,
        num_node_tuples,
        rmsd_to_crystals,
        pdbids,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][(spatial_poses[idx] >= spatial_pos_max) & (spatial_poses[idx] != 511)] = 0.0
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    if not has_all_poses:
        all_poses = None
    else:
        all_poses = torch.cat(
            [pad_all_poses_unsqueeze(i, max_node_num) for i in all_poseses] if all_poseses is not None else None
        )  # workaround for avoid auto adding 1 to pos
    pos = torch.cat(
        [pad_pos_unsqueeze(i, max_node_num) for i in poss]
    )
    crystal_pos = torch.cat(
        [pad_pos_unsqueeze(i, max_node_num) for i in crystal_poss]
    )  # workaround for avoid auto adding 1 to pos
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    lnode = torch.tensor([i[0] for i in num_node_tuples])
    pnode = torch.tensor([i[1] for i in num_node_tuples])
    allnode = torch.tensor([i[0] + i[1] for i in num_node_tuples])
    rmsd_to_crystal = torch.tensor(rmsd_to_crystals)

    ret = dict(
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        all_poses=all_poses,
        pos=pos,
        crystal_pos=crystal_pos,
        edge_input=edge_input,
        rmsd_to_crystal=rmsd_to_crystal,
        lnode=lnode,
        pnode=pnode,
        allnode=allnode,
        pdbid = pdbids,
    )

    # remove none items
    ret = {k: v for k, v in ret.items() if v is not None}

    return ret


def build_md_kde_dm(data_path, reweighting_file,need_all_poses=False):
    return MDKDEEvalDatasetDM(data_path, reweighting_file,need_all_poses=need_all_poses)


def build_md_kde_dataset(raw_dataset):
    return MDKDEEvalDataset(raw_dataset)
