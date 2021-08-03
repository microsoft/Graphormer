# Benchmarking Graph Neural Networks

[https://arxiv.org/abs/2003.00982](https://arxiv.org/abs/2003.00982)

[https://github.com/graphdeeplearning/benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns)

## Results

#### ZINC-500K
Method        | #params | test MAE   |
--------------|---------|------------|
GIN          | 509.5K  | 0.526     |
GraphSage          | 505.3K  | 0.398      |
GAT          | 531.3K  | 0.384      |
GCN          | 505.1K  | 0.367      |
GT          | 588.9K  | 0.226      |
GatedGCN-PE          | 505.0K  | 0.214      |
MPNN (sum)          | 480.8K  | 0.145      |
PNA          | 387.2K  | 0.142      |
SAN          | 508.6K  | 0.139      |
Graphormer-Slim   | 489.3K  | **0.122**      |

## Example Usage

```
[ -z "${exp_name}" ] && exp_name="zinc"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 80 --hidden_dim 80 --num_heads 8 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20"
[ -z "${warmup_updates}" ] && warmup_updates="40000"
[ -z "${tot_updates}" ] && tot_updates="400000"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "exp_name: ${exp_name}"
echo "warmup_updates: ${warmup_updates}"
echo "tot_updates: ${tot_updates}"
echo "==============================================================================="

save_path="../../exps/zinc/$exp_name-$warmup_updates-$tot_updates/$seed"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0 \
      python ../../graphormer/entry.py --num_workers 8 --seed $seed --batch_size 256 \
      --dataset_name ZINC \
      --gpus 1 --accelerator ddp --precision 16 \
      $arch \
      --check_val_every_n_epoch 10 --warmup_updates $warmup_updates --tot_updates $tot_updates \
      --default_root_dir $save_path
```

## Citation
Please kindly cite this paper if you use the code:
```
@article{ying2021transformers,
  title={Do Transformers Really Perform Bad for Graph Representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2106.05234},
  year={2021}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
