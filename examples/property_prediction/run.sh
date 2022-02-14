#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

git_commit=$(git rev-parse HEAD)
echo "Using git commit ${git_commit}"
checkpoint_dir=checkpoint_$1_$2_$3_$4_${git_commit}
mkdir checkpoint_$1_$2_$3_$4_${git_commit}

if [[ $4 == "fp16" ]]; then
    if [[ $3 == "graphormer_v2" ]] && [[ $1 == "large" ]] && [[ $2 == "pcqm4mv2" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1 \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 4e-5 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --fp16 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    elif [[ $3 == "graphormer_v2" ]] && [[ $1 == "large" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1 \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 8e-5 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --fp16 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    elif [[ $3 == "graphormer_v2" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1 \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --fp16 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    elif [[ $3 == "graphormer_v1" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1_v1 \
        --num-classes 1 \
        --attention-dropout 0.1 --input-dropout 0.0 --dropout 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --fp16 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    fi
else
    if [[ $3 == "graphormer_v2" ]] && [[ $1 == "large" ]] && [[ $2 == "pcqm4mv2" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1 \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 4e-5 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    elif [[ $3 == "graphormer_v2" ]] && [[ $1 == "large" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1 \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 8e-5 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    elif [[ $3 == "graphormer_v2" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1 \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    elif [[ $3 == "graphormer_v1" ]]; then
        fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name $2 \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_$1_v1 \
        --num-classes 1 \
        --attention-dropout 0.1 --input-dropout 0.0 --dropout 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --data-buffer-size 20 \
        --max-epoch 300 \
        --save-dir /blob/mol/${checkpoint_dir}
    fi
fi
