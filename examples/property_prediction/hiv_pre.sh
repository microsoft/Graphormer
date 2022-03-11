#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

n_gpu=1
epoch=4
max_epoch=$((epoch + 1))
batch_size=128
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates*16/100))

CUDA_VISIBLE_DEVICES=3 fairseq-train \
--user-dir ../../graphormer \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name ogbg-molhiv \
--dataset-source ogb \
--task graph_prediction_with_flag \
--criterion binary_logloss_with_flag \
--arch graphormer_base \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
--lr 2e-4 --end-learning-rate 1e-5 \
--batch-size $batch_size \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 768 \
--encoder-ffn-embed-dim 768 \
--encoder-attention-heads 32 \
--max-epoch $max_epoch \
--save-dir ./ckpts \
--pretrained-model-name pcqm4mv1_graphormer_base_for_molhiv \
--seed 1 \
--flag-m 3 \
--flag-step-size 0.01 \
--flag-mag 0 \
--pre-layernorm
