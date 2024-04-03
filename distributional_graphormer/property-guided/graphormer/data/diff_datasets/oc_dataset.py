from .diff_dataset import CrossDockedDM, CrossDockedDataset, CrossDockedRawDataset
from .diff_dataset import crossdocked_batch_collater

from ..collator import (
    pad_1d_unsqueeze,
    pad_2d_unsqueeze,
    pad_all_poses_unsqueeze,
    pad_pos_unsqueeze,
    pad_3d_unsqueeze,
    pad_attn_bias_unsqueeze,
    pad_spatial_pos_unsqueeze,
)

import torch

class OCRawDataset(CrossDockedRawDataset):
    def __init__(self, path, file_list, split, subdir="full"):
        super().__init__(path, file_list, subdir)
        self.subdir = subdir
        self.split = split

class OCKDEEvalDatasetDM(CrossDockedDM):
    def get_split(self, split):
        if split not in self.split_lut:
            split_list_lines = open(self.path + f"/{split}.list", "r").readlines()
            split_list = self.process_file_list(split_list_lines)
            if split.endswith("-all-poses") or split.find("-all-poses.") != -1:
                self.split_lut[split] = OCRawDataset(self.path, split_list, split, "all-poses")
            else:
                self.split_lut[split] = OCRawDataset(self.path, split_list, split, "all")
        return self.split_lut[split]

class OCKDEEvalDataset(CrossDockedDataset):
    def collater(self, samples):
        if self.raw_dataset.split.find("all-poses") != -1:
            return oc_batch_collater(samples, has_all_poses=True)
        else:
            return oc_batch_collater(samples, has_all_poses=False)

def oc_batch_collater(
    items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, has_all_poses=False
):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.attn_bias,
            item.x,
            torch.cat([pos.unsqueeze(0) for pos in item.all_poses], axis=0) if has_all_poses else None,
            item.pos,
            item.tags,
            item.natoms,
            item.lnode,
            item.cell.unsqueeze(0),
            item.atomic_numbers,
            item.sid,
            item.radius,
        )
        for item in items
    ]
    (
        attn_biases,
        xs,
        all_poseses,
        poses,
        tagss,
        natomss,
        lnodess,
        cells,
        atomic_numberss,
        sids,
        radiuss,
    ) = zip(*items)

    max_node_num = max(i.size(0) for i in xs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    if not has_all_poses:
        all_poses = None
    else:
        all_poses = torch.cat(
            [pad_all_poses_unsqueeze(i, max_node_num) for i in all_poseses] if all_poseses is not None else None
        )  # workaround for avoid auto adding 1 to pos
    pos = torch.cat(
        [pad_pos_unsqueeze(i, max_node_num) for i in poses]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    sid = torch.tensor([sid for sid in sids], dtype=torch.long)

    tags = torch.cat(
        [pad_1d_unsqueeze(i, max_node_num) for i in tagss]
    )

    radius = torch.tensor([radius for radius in radiuss], dtype=torch.long)

    natoms = torch.tensor(list(natomss))
    lnodes = torch.tensor(list(lnodess))
    cell = torch.cat([i for i in cells], axis=0)
    atomic_numbers = torch.cat([i for i in atomic_numberss], axis=0)

    ret = dict(
        attn_bias=attn_bias,
        x=x,
        all_poses=all_poses,
        pos=pos,
        tags=tags,
        natoms=natoms,
        lnodes=lnodes,
        cell=cell,
        atomic_numbers = atomic_numbers,
        sid=sid,
        radius=radius,
    )

    # remove none items
    ret = {k: v for k, v in ret.items() if v is not None}

    return ret


def build_oc_kde_dm(data_path):
    return OCKDEEvalDatasetDM(data_path)


def build_oc_kde_dataset(raw_dataset):
    return OCKDEEvalDataset(raw_dataset)
