# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#!/usr/bin/env bash

[ -z "${exp_name}" ] && exp_name="hiv_flag"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 768 --hidden_dim 768 --intput_dropout_rate 0.0 --attention_dropout_rate 0.1 --dropout_rate 0.1 --weight_decay 0.0 --n_layers 12 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${batch_size}" ] && batch_size="128"         # Alternatively, you can decrease the bsz to 64 and use 2 GPUs, if you do not have 32G GPU memory.
[ -z "${epoch}" ] && epoch="8"
[ -z "${peak_lr}" ] && peak_lr="2e-4"
[ -z "${end_lr}" ] && end_lr="1e-9"

[ -z "${flag_m}" ] && flag_m="2"
[ -z "${flag_step_size}" ] && flag_step_size="0.2"
[ -z "${flag_mag}" ] && flag_mag="0"

[ -z "${ckpt_path}" ] && ckpt_path="../../checkpoints/hiv/<your_pretrained_model_for_hiv>"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "ckpt_path ${ckpt_path}"
echo "arch: ${arch}"
echo "batch_size: ${batch_size}"
echo "peak_lr ${peak_lr}"
echo "end_lr ${end_lr}"
echo "flag_m ${flag_m}"
echo "flag_step_size :${flag_step_size}"
echo "flag_mag: ${flag_mag}"
echo "seed: ${seed}"
echo "epoch: ${epoch}"
echo "==============================================================================="

n_gpu=1                   # Please use 1 GPU (We use 1 32GB V100 card) to reproduce our results.
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates/10))
max_epochs=$((epoch+1))
echo "=====================================ARGS======================================"
echo "tot_updates ${tot_updates}"
echo "warmup_updates: ${warmup_updates}"
echo "max_epochs: ${max_epochs}"
echo "==============================================================================="

default_root_dir=../../exps/hiv/$exp_name/$seed
mkdir -p $default_root_dir

python ../../graphormer/entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name ogbg-molhiv \
      --gpus $n_gpu --accelerator ddp --precision 16 $arch \
      --default_root_dir $default_root_dir \
      --tot_updates $tot_updates --warmup_updates $warmup_updates --max_epochs $max_epochs \
      --checkpoint_path $ckpt_path \
      --peak_lr $peak_lr --end_lr $end_lr --progress_bar_refresh_rate 10 \
      --flag --flag_m $flag_m --flag_step_size $flag_step_size --flag_mag $flag_mag


# validate and test on every checkpoint
checkpoint_dir=$default_root_dir/lightning_logs/checkpoints/
echo "=====================================EVAL======================================"
for file in `ls $checkpoint_dir/*.ckpt`
do
      echo -e "\n\n\n ckpt:"
      echo "$file"
      echo -e "\n\n\n"
      python ../../graphormer/entry.py --num_workers 8 --seed 1 --batch_size $batch_size \
            --dataset_name ogbg-molhiv \
            --gpus 1 --accelerator ddp --precision 16 $arch \
            --default_root_dir tmp/ \
            --checkpoint_path $file --validate --progress_bar_refresh_rate 100

      python ../../graphormer/entry.py --num_workers 8 --seed 1 --batch_size $batch_size \
            --dataset_name ogbg-molhiv \
            --gpus 1 --accelerator ddp --precision 16 $arch \
            --default_root_dir tmp/ \
            --checkpoint_path $file --test --progress_bar_refresh_rate 100
done
echo "==============================================================================="
