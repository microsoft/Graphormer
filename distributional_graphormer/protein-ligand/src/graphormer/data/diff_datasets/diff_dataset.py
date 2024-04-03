import torch
import pickle
import numpy as np
import lmdb
from fairseq.data import BaseWrapperDataset, FairseqDataset
from fairseq.data import data_utils
from scipy.spatial.transform import Rotation
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
    def __init__(self, path, reweighting_file, split, file_list, train_subset, need_all_poses=False,subdir="full"):
        super().__init__()
        self.path = path
        self.subdir = subdir
        self.split = split
        self.file_list = file_list
        self.reweighting_file = reweighting_file
        self.train_subset = train_subset
        self.need_all_poses = need_all_poses
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

    def set_dtype(self, data):
        data.x = data.x.type(torch.long)
        data.edge_index = data.edge_index.type(torch.long)
        data.edge_attr = data.edge_attr.type(torch.long)
        # data.pos = data.pos.float()
        return data

    def add_crystal_pos(self, data, crystal_data):
        '''
        add crystal pos to data
        '''
        # mask out ligand atoms in crystal pos to avoid data leakage
        # crystal_data.pos[0: crystal_data.num_node[0]] = 0
        data.crystal_pos = crystal_data.pos
        data.x = crystal_data.x
        data.edge_index = crystal_data.edge_index
        data.edge_attr = crystal_data.edge_attr

        # # apply a random rotation matrix to crystal pos
        # R = torch.tensor(Rotation.random().as_matrix()).float()
        # data.crystal_pos = torch.matmul(data.crystal_pos, R)

        return data

    def add_all_poses(self, data, pdbid):
        # add all poses to data
        poses = []
        for i in range(0,100):
            pose = self.txn.get((pdbid+'_'+str(i)).encode())
            if pose is not None:
                pose = pickle.loads(pose)
                poses.append(pose.pos)
        data.all_poses = poses
        return data

    def __getitem__(self, index):
        if self.txn is None:
            self.build_dataset()
        data = pickle.loads(self.txn.get(self.file_list[index].encode()))

        if self.need_all_poses:
            data = self.add_all_poses(data,self.file_list[index].split('_')[0])

        crystal = self.file_list[index].rsplit('_',1)[0]+'_crystal'
        crystal = self.txn.get(crystal.encode())
        # determine whether crystal is in lmdb
        if crystal is not None:
            crystal_data = pickle.loads(crystal)
            data = self.add_crystal_pos(data,crystal_data)
        return self.set_dtype(data)

    def get_rmsd_to_crystal(self, index):

        if self.split in self.train_subset:
            if self.reweighting_file != "":
                if not hasattr(self, "rmsd_lut"):
                    print(f'finding rmsd reweighting file {self.reweighting_file} ...')
                    # check if we have a RMSD file, or return 0.0
                    try:
                        file_list  = open(self.reweighting_file)
                        new_file_list = [float(file.split(',')[1].strip()) for file in file_list]
                        self.rmsd_lut = new_file_list
                    except FileNotFoundError:
                        print('no rmsd reweighting file found, using 0.0')
                        self.reweighting_file = ""
                        self.rmsd_lut = None
                        return 0.0
                return self.rmsd_lut[index]
            else:
                if not hasattr(self, "rmsd_lut"):
                    print("no rmsd reweighting needed, using 0.0")
                    self.rmsd_lut = None
        return 0.0


    def __len__(self):
        return len(self.file_list)


class CrossDockedDM:
    def __init__(self, path, reweighting_file,need_all_poses=False):
        super().__init__()
        self.path = path
        self.reweighting_file = reweighting_file
        self.split_lut = {}
        self.need_all_poses = need_all_poses

    def process_file_list(self, file_list):
        new_file_list = []
        for file in file_list:
            if file:
                new_file_list.append(file.strip())
        return new_file_list

    def get_split(self, split, train_subset):
        if split not in self.split_lut:
            split_list_lines = open(self.path + f"/{split}.list", "r").readlines()
            split_list = self.process_file_list(split_list_lines)
            self.split_lut[split] = CrossDockedRawDataset(self.path, self.reweighting_file, split, split_list, train_subset,self.need_all_poses)
        return self.split_lut[split]


class CrossDockedDataset(FairseqDataset):
    def __init__(self, raw_dataset):
        super().__init__()
        self.raw_dataset = raw_dataset

    def __getitem__(self, index):
        item = self.raw_dataset[index].clone()
        ret_item = preprocess_item(item)
        ret_item.rmsd_to_crystal = self.raw_dataset.get_rmsd_to_crystal(index)
        return ret_item

    def __len__(self):
        return len(self.raw_dataset)

    def collater(self, samples):
        return crossdocked_batch_collater(samples)


def crossdocked_batch_collater(
    items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20
):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    # if crystal_pos not in items keys, set a flag
    if 'crystal_pos' not in items[0].keys:
        crystal_pos_flag = False
    else:
        crystal_pos_flag = True
    items = [
        (
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.pos,
            item.crystal_pos if crystal_pos_flag else torch.rand_like(item.pos),
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.num_node,
            item.rmsd_to_crystal,
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
        crystal_poss,
        edge_inputs,
        num_node_tuples,
        rmsd_to_crystals,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][(spatial_poses[idx] >= spatial_pos_max) & (spatial_poses[idx] != 511)] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    pos = torch.cat(
        [pad_pos_unsqueeze(i, max_node_num) for i in poss]
    )  # workaround for avoid auto adding 1 to pos
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
        pos=pos,
        crystal_pos=crystal_pos,
        edge_input=edge_input,
        rmsd_to_crystal=rmsd_to_crystal,
        lnode=lnode,
        pnode=pnode,
        allnode=allnode,
    )

    # remove none items
    ret = {k: v for k, v in ret.items() if v is not None}

    return ret


def build_diff_dm(data_path,reweighting_file):
    return CrossDockedDM(data_path,reweighting_file)


def build_diff_dataset(raw_dataset):
    return CrossDockedDataset(raw_dataset)
