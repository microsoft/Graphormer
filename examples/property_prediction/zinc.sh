# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 fairseq-train \
--user-dir ../../graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name zinc \
--dataset-source pyg \
--task graph_prediction \
--criterion l1_loss \
--arch graphormer_slim \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 400000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 64 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 80 \
--encoder-ffn-embed-dim 80 \
--encoder-attention-heads 8 \
--max-epoch 10000 \
--save-dir ./ckpts
