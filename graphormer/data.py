# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collator import collator
from wrapper import MyGraphPropPredDataset, MyPygPCQM4MDataset, MyZINCDataset

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ogb
import ogb.lsc
import ogb.graphproppred
from functools import partial


dataset = None


def get_dataset(dataset_name='abaaba'):
    global dataset
    if dataset is not None:
        return dataset

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))
    if dataset_name == 'ogbg-molpcba':
        dataset = {
            'num_class': 128,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'ap',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molpcba'),
            'dataset': MyGraphPropPredDataset('ogbg-molpcba', root='../../dataset'),
            'max_node': 128,
        }
    elif dataset_name == 'ogbg-molhiv':
        dataset = {
            'num_class': 1,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molhiv'),
            'dataset': MyGraphPropPredDataset('ogbg-molhiv', root='../../dataset'),
            'max_node': 128,
        }
    elif dataset_name == 'PCQM4M-LSC':
        dataset = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': ogb.lsc.PCQM4MEvaluator(),
            'dataset': MyPygPCQM4MDataset(root='../../dataset'),
            'max_node': 128,
        }
    elif dataset_name == 'ZINC':
        dataset = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': ogb.lsc.PCQM4MEvaluator(),  # same objective function, so reuse it
            'train_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='train'),
            'valid_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='val'),
            'test_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='test'),
            'max_node': 128,
        }
    else:
        raise NotImplementedError

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset


class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
        self,
        dataset_name: str = 'ogbg-molpcba',
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def setup(self, stage: str = None):
        if self.dataset_name == 'ZINC':
            self.dataset_train = self.dataset['train_dataset']
            self.dataset_val = self.dataset['valid_dataset']
            self.dataset_test = self.dataset['test_dataset']
        else:
            split_idx = self.dataset['dataset'].get_idx_split()
            self.dataset_train = self.dataset['dataset'][split_idx["train"]]
            self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
            self.dataset_test = self.dataset['dataset'][split_idx["test"]]

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(test_dataloader)', len(loader))
        return loader
