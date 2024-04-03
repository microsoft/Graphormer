import torch
import pickle
import numpy as np
import lmdb
from fairseq.data import BaseWrapperDataset, FairseqDataset
from fairseq.data import data_utils

# from torch_geometric.data import Data

from ..wrapper import preprocess_item
from ..collator import (
    pad_1d_unsqueeze,
    pad_2d_unsqueeze,
    pad_pos_unsqueeze,
    pad_3d_unsqueeze,
    pad_attn_bias_unsqueeze,
    pad_edge_type_unsqueeze,
    pad_spatial_pos_unsqueeze,
)

import torch

class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed):
        super().__init__(dataset)
        self.num_samples = len(dataset)
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        with data_utils.numpy_seed((self.seed, epoch - 1)):
            self.sort_order = np.random.permutation(self.num_samples)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False


class CrossDockedRawDataset(FairseqDataset):
    def __init__(self, path, file_list, subdir="full"):
        super().__init__()
        self.path = path
        self.subdir = subdir
        self.file_list = file_list
        self.env = None
        self.txn = None

    def build_dataset(self):
        self.env = lmdb.open(
            self.path + "/" + self.subdir,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

        self.init_pos_env = lmdb.open(self.path + "/is2re-with-bonds")
        self.init_pos_txn = self.init_pos_env.begin(write=False)

    def set_dtype(self, data):
        data.atomic_numbers = data.atomic_numbers.type(torch.long)
        return data

    def rmsd(self, pos1, pos2):
        _, indices1 = torch.sort(torch.sum(pos1, dim=-1), stable=True)
        _, indices2 = torch.sort(torch.sum(pos2, dim=-1), stable=True)
        return torch.sqrt(torch.mean(torch.sum((pos1[indices1] - pos2[indices2]) ** 2, dim=-1), dim=-1))

    def __getitem__(self, index):
        if self.txn is None:
            self.build_dataset()
        data = pickle.loads(self.txn.get(self.file_list[index].encode()))
        sid = data.sid
        byte = self.init_pos_txn.get(f"{sid}".encode())
        if byte is not None:
            init_pos_graph = pickle.loads(byte)
            data.init_pos = init_pos_graph.pos
            data.bonds = init_pos_graph.bonds
        else:
            raise ValueError(f"SID {sid} not found in is2rs data base")
        return self.set_dtype(data)

    def __len__(self):
        return len(self.file_list)


class CrossDockedDM:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.split_lut = {}

    def process_file_list(self, file_list):
        new_file_list = []
        for file in file_list:
            if file:
                new_file_list.append(file.strip())
        return new_file_list

    def get_split(self, split):
        if split not in self.split_lut:
            split_list_lines = open(self.path + f"/{split}.list", "r").readlines()
            split_list = self.process_file_list(split_list_lines)
            self.split_lut[split] = CrossDockedRawDataset(self.path, split_list)
        return self.split_lut[split]


class CrossDockedDataset(FairseqDataset):
    def __init__(self, raw_dataset):
        super().__init__()
        self.raw_dataset = raw_dataset

    def __getitem__(self, index):
        item = self.raw_dataset[index].clone()
        return preprocess_item(item)

    def __len__(self):
        return len(self.raw_dataset)

    def collater(self, samples):
        return crossdocked_batch_collater(samples)


def crossdocked_batch_collater(
    items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20
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
            item.pos,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.num_node,
            item.tags,
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
        poss,
        edge_inputs,
        num_node_tuples,
        tagss,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][(spatial_poses[idx] >= spatial_pos_max) & (spatial_poses[idx] != 511)] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    pos = torch.cat(
        [pad_pos_unsqueeze(i, max_node_num) for i in poss]
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

    tags = torch.cat(
        [pad_1d_unsqueeze(i, max_node_num) for i in tagss]
    )

    ret = dict(
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        pos=pos,
        edge_input=edge_input,
        lnode=lnode,
        pnode=pnode,
        allnode=allnode,
        tags=tags,
    )

    # remove none items
    ret = {k: v for k, v in ret.items() if v is not None}

    return ret


def build_diff_dm(data_path):
    return CrossDockedDM(data_path)


def build_diff_dataset(raw_dataset):
    return CrossDockedDataset(raw_dataset)
