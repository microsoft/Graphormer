# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import os.path as osp
import shutil
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset, Data

class PCQv2PYG(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph = smiles2graph, transform=None, pre_transform=None):
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.original_root = root                        
        self.smiles2graph = smiles2graph
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) 

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(self.url, self.original_root)
        extract_zip(path, self.original_root)
        os.unlink(path)
        
    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = df['smiles'][:20000]
        homolumogap_list = df['homolumogap'][:20000]
        data_list = []

        print("Converting SMILES strings to graphs...")
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            graph = self.smiles2graph(smiles_list[i])
            homolumogap = homolumogap_list[i]
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            data_list.append(data)
        
        # double check NaN values
        # split_dict = self.get_idx_split()
        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        # split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        # return split_dict
        split_dict = {'train': None, 'valid': None, 'test-dev': None}
        split_dict['train'] = torch.from_numpy(np.arange(0, 16000)).to(torch.int64)
        split_dict['valid'] = torch.from_numpy(np.arange(16000, 18000)).to(torch.int64)
        split_dict['test-dev'] = torch.from_numpy(np.arange(18000, 20000)).to(torch.int64)
        return split_dict

if __name__ == '__main__':
    dataset = PCQv2PYG()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())