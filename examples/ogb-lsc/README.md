# Open Graph Benchmark - Large-Scale Challenge (KDD Cup 2021)

[https://ogb.stanford.edu/kddcup2021/](https://ogb.stanford.edu/kddcup2021/)

[https://arxiv.org/abs/2103.09430](https://arxiv.org/abs/2103.09430)

## Results

#### PCQM4M-LSC
Method        | #params | train MAE | valid MAE |
--------------|---------|-----------|-----------|
GCN          | 2.0M    | 0.1318    | 0.1691    |
GIN          | 3.8M    | 0.1203    | 0.1537    |
GCN-VN          | 4.9M    | 0.1225    | 0.1485    |
GIN-VN          | 6.7M    | 0.1150    | 0.1395    |
Graphormer-Small| 12.5M   | 0.0778    | 0.1264    |
Graphormer   | 47.1M   | 0.0582    | **0.1234**    |

## Example Usage

```
[ -z "${exp_name}" ] && exp_name="pcq"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 768 --hidden_dim 768 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${batch_size}" ] && batch_size="256"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "==============================================================================="

default_root_dir="../../exps/pcq/$exp_name/$seed"
mkdir -p $default_root_dir
n_gpu=$(nvidia-smi -L | wc -l)

python ../../graphormer/entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name PCQM4M-LSC \
      --gpus $n_gpu --accelerator ddp --precision 16 --gradient_clip_val 5.0 \
      $arch \
      --default_root_dir $default_root_dir
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
