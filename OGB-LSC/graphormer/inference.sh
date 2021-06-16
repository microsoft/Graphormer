# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash

python src/entry.py --num_workers 8 --seed 1 --batch_size 256 \
      --dataset_name PCQM4M-LSC \
      --gpus 1 --accelerator ddp --precision 32 $arch \
      --default_root_dir tmp/ \
      --checkpoint_path $ckpt_path/$ckpt_name --test --progress_bar_refresh_rate 100


# copy to target
mkdir -p logits/$ckpt_name
cp y_pred.pt logits/$ckpt_name/y_pred.pt
