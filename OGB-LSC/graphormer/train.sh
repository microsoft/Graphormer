# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"      # all_fold
echo "arch: ${arch}"              # --ffn_dim 768 --hidden_dim 768 --attention_dropout_rate 0.1 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20 --weight_decay 0.0 --intput_dropout_rate 0.0 --warmup_updates 10000 --tot_updates 1500000
echo "seed: ${seed}"              # 0
echo "batch_size: ${batch_size}"  # 256 x 4
echo "==============================================================================="

default_root_dir=$exp_name/$seed
mkdir -p $default_root_dir
n_gpu=$(nvidia-smi -L | wc -l)

python src/entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name PCQM4M-LSC \
      --gpus $n_gpu --accelerator ddp --precision 16 --gradient_clip_val 5.0 \
      $arch \
      --default_root_dir $default_root_dir --progress_bar_refresh_rate 100
