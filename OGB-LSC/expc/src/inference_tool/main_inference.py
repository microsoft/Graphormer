# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

### importing OGB-LSC
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import torch
from torch_geometric.data.dataloader import DataLoader

import os
import numpy as np
import random
from tqdm import tqdm

import sys
sys.path.append('.')

from src.model import Net
from src.inference_tool.config import process_config

import time


reg_criterion = torch.nn.L1Loss()


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def main():
    config = process_config(sys.argv)

    if config.get('seed') is not None:
        print("seed: ", config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PygPCQM4MDataset(root=config.dataset)
    split_idx = dataset.get_idx_split()
    dataset_test = dataset[split_idx["test"]]
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    net = Net
    model = net(config.architecture).to(device)

    print("Loading model from {}...".format(config.checkpoint), end=' ')
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded.")

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    print('Generate inference logits on test data...')

    time_start = time.time()
    y_pred = test(model, device, test_loader)
    time_len = time.time() - time_start
    print("Spend " + str(time_len) + " to inference_tool " + str(len(dataset_test)) + " moles.")
    print("Average inference_tool time: " + str(time_len / len(dataset_test)) + " per mole.")

    save_inference_logits(config.output, 'y_pred', y_pred)


def save_inference_logits(dir, file, y_pred):
    print('Saving test submission file...')
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, file)
    assert (isinstance(filename, str))
    assert (isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))
    assert (y_pred.shape == (377423,))

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    y_pred = y_pred.astype(np.float32)
    np.savez_compressed(filename, y_pred=y_pred)


if __name__ == "__main__":
    main()
