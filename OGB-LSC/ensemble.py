# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import torch
from ogb.lsc import PCQM4MEvaluator


gf_all_s0 = torch.load('graphormer/logits/all_fold_seed0.ckpt/y_pred.pt')
gf_all_s1 = torch.load('graphormer/logits/all_fold_seed1.ckpt/y_pred.pt')

gf_f0= torch.load('graphormer/logits/fold0.ckpt/y_pred.pt')
gf_f1= torch.load('graphormer/logits/fold1.ckpt/y_pred.pt')
gf_f2= torch.load('graphormer/logits/fold2.ckpt/y_pred.pt')
gf_f3= torch.load('graphormer/logits/fold3.ckpt/y_pred.pt')
gf_f4= torch.load('graphormer/logits/fold4.ckpt/y_pred.pt')
gf_f5= torch.load('graphormer/logits/fold5.ckpt/y_pred.pt')
gf_f6= torch.load('graphormer/logits/fold6.ckpt/y_pred.pt')
gf_f7= torch.load('graphormer/logits/fold7.ckpt/y_pred.pt')

expc_f0 = np.load('expc/logits/fold0.ckpt/y_pred.npz')['y_pred']
expc_f1 = np.load('expc/logits/fold1.ckpt/y_pred.npz')['y_pred']
expc_f2 = np.load('expc/logits/fold2.ckpt/y_pred.npz')['y_pred']
expc_f3 = np.load('expc/logits/fold3.ckpt/y_pred.npz')['y_pred']
expc_f4 = np.load('expc/logits/fold4.ckpt/y_pred.npz')['y_pred']
expc_f5 = np.load('expc/logits/fold5.ckpt/y_pred.npz')['y_pred']
expc_f6 = np.load('expc/logits/fold6.ckpt/y_pred.npz')['y_pred']
expc_f7 = np.load('expc/logits/fold7.ckpt/y_pred.npz')['y_pred']

all_logits = np.vstack([
    gf_all_s0, gf_all_s1,
    gf_f0, gf_f1, gf_f2, gf_f3, gf_f4, gf_f5, gf_f6, gf_f7,
    expc_f0, expc_f1, expc_f2, expc_f3,
    expc_f4, expc_f5, expc_f6, expc_f7,
])

weights =  np.array([
    0.05, 0.08,
    0.05, 0.05, 0.05, 0.08, 0.05, 0.08, 0.08, 0.05,
    0.05, 0.05, 0.05, 0.03, 0.05, 0.03, 0.05, 0.03,
])

w_logits = np.zeros([377423])
for idx, w in enumerate(weights):
    w_logits += all_logits[idx, :] * w
w_logits /= sum(weights)
PCQM4MEvaluator().save_test_submission({'y_pred': w_logits}, '0609')
