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

import tarfile
import jsonlines
import gzip


# few-shot not implemented yet
class FSmolPYG(InMemoryDataset):
    def __init__(
        self, 
        root, 
        split, 
        seed: int = 0,
        transform=None, 
        pre_transform=None
    ) -> None:
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.original_root = root       
        self.url = 'https://figshare.com/ndownloader/files/31345321'
        self.task_to_head = {}
        self.num_heads_total = 0
        super().__init__(root, transform, pre_transform)

        # calculate the total number of heads & construct the task_to_head dict
        # path: datasets/fsmol/raw/train
        path = osp.join(self.raw_dir(), split)
        self.unzip_gz_and_calc_head(path)        
        self.data, self.slices = torch.load(self.processed_file_names)
      
    def unzip_gz_and_calc_head(self, path):
        if os.path.exists(path):
            head_num = 0
            dirs = os.listdir(path)
            for dir in dirs:
                if '.gz' in dir:
                    filename = dir.replace(".gz","")
                    assert filename not in self.task_to_head, f"Duplicated task {filename} in split {self.split}!"
                    self.task_to_head['filename'] = head_num
                    head_num += 1
                    gzip_file = gzip.GzipFile(path + dir)
                    with open(path + filename,'wb+') as f:
                        f.write(gzip_file.read())
            for dir in dirs:  # delete .gz files
                if '.gz' in dir:
                    os.unlink(dir)
            self.num_heads_total = head_num
        else:
            raise Exception("The file to unzip does not exist!")
          
    @property
    def raw_dir(self):  # datasets/fsmol/raw/train
        return f"{self.root}/fsmol/raw"

    @property
    def processed_dir(self):
        return f"{self.root}/fsmol/processed"

    @property
    def raw_file_names(self):
        return 'fsmol.tar'

    @property
    def processed_file_names(self):
        return f'{self.split}.pt'

    def download(self):
        # Download fsmol.tar to `self.raw_dir` & unzip the file.
        # datasets/raw/fsmol/train
        path = download_url(self.url, self.original_root)
        tar = tarfile.open(path)
        tar.extractall()
        tar.close()
        # os.unlink(path)  # keep the tar file
        
    def process(self):
        # Read data into huge `Data` list.
        path = osp.join(self.raw_dir(), self.split)
        data_list = []
        dirs = os.listdir(path)
        for dir in dirs:            
            with open(dir, "r+", encoding="utf8") as f:
                filename = dir.replace(".gz","")
                head = self.task_to_head[filename]
                for item in jsonlines.Reader(f):
                    data = Data()
                    data.head = head
                    data.smiles = item["SMILES"]
                    data.y = -1 if item["Property"] == 0.0 else head
                    data_list.append(data)

        print(f"Converting SMILES strings to graphs in split '{self.split}':")
        for i, data in enumerate(tqdm(data_list)):
            graph = self.smiles2graph(data.smiles)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            del data.smiles

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = FSmolPYG()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())