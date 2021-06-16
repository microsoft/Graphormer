# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

### importing OGB-LSC
from ogb.lsc import PCQM4MEvaluator
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from torch_geometric.data.dataloader import DataLoader

import os
import numpy as np
import random

import sys
sys.path.append('.')

from model import Net
from utils.config import process_config, get_args

from sklearn.model_selection import KFold


reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer, config):
    model.train()
    loss_accum = 0

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)

        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()
    return loss_accum / len(loader)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def main():
    args = get_args()
    config = process_config(args)
    print(config)

    if config.get('seed') is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PygPCQM4MDataset(root='dataset/')

    evaluator = PCQM4MEvaluator()

    kf = KFold(n_splits=8, shuffle=True, random_state=10086)
    split_idx = dataset.get_idx_split()
    train_val = np.hstack([split_idx["train"], split_idx["valid"]])
    train_vals = [i for i in kf.split(train_val)]
    train_split = train_vals[config.fold][0].tolist()
    valid_split = train_vals[config.fold][1].tolist()

    dataset_train = dataset[train_split]
    dataset_val = dataset[valid_split]
    dataset_test = dataset

    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)

    valid_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers = config.num_workers)

    if config.get('save_test_dir') is not None:
        test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers = config.num_workers)

    net = Net
    model = net(config.architecture).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_valid_mae = 1000

    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.decay_rate)

    writer = SummaryWriter(log_dir=config.directory)
    ts_algo_hp = str(config.time_stamp) + '_' \
                 + str(config.commit_id[0:7]) + '_' \
                 + str(config.architecture.exp_n) \
                 + str(config.architecture.exp_nonlinear) \
                 + str(config.architecture.exp_bn) \
                 + str(config.architecture.pooling) \
                 + str(config.architecture.JK) \
                 + str(config.architecture.layers) + '_' \
                 + str(config.architecture.hidden) + '_' \
                 + str(config.architecture.dropout) + '_' \
                 + str(config.learning_rate) + '_' \
                 + str(config.step_size) + '_' \
                 + str(config.decay_rate) + '_' \
                 + 'B' + str(config.batch_size) \
                 + 'S' + str(config.get('seed', 'na')) \
                 + 'W' + str(config.get('num_workers', 'na'))\
                 + 'f' + str(config.fold)

    for epoch in range(1, config.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer, config)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae})

        lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalars('pcqm4m', {ts_algo_hp + '/lr': lr}, epoch)
        # writer.add_scalars('pcqm4m', {ts_algo_hp + '/te': test_error}, epoch)
        writer.add_scalars('pcqm4m', {ts_algo_hp + '/ve': valid_mae}, epoch)
        writer.add_scalars('pcqm4m', {ts_algo_hp + '/ls': train_mae}, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if config.get('checkpoint_dir') is not None:
                print('Saving checkpoint...')
                os.makedirs(config.get('checkpoint_dir'), exist_ok=True)
                checkpoint = {'model_state_dict': model.state_dict()}
                torch.save(checkpoint, os.path.join(config.get('checkpoint_dir'), 'pcqm4m_' + ts_algo_hp + '_checkpoint.pt'))

            if config.get('save_test_dir') is not None:
                print('Predicting on test data...')
                y_pred = test(model, device, test_loader)
                save_inference_logits(config.get('save_test_dir'), 'all_y_pred_pcqm4m' + ts_algo_hp, y_pred)


        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')

    writer.close()


def save_inference_logits(dir, file, y_pred):
    print('Saving test submission file...')
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, file)
    assert (isinstance(filename, str))
    assert (isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))
    # assert (y_pred.shape == (len(dataset),))

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    y_pred = y_pred.astype(np.float32)
    np.savez_compressed(filename, y_pred=y_pred)

    # evaluator.save_test_submission({'y_pred': y_pred}, config.get('save_test_dir'))


if __name__ == "__main__":
    main()
