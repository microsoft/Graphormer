## Open Graph Benchmark - Large-Scale Challenge (KDD Cup 2021)

The detailed description of the dataset could be found [here](https://ogb.stanford.edu/kddcup2021/).

### Example Usage
To train Graphormer-Base on PCQM4Mv1 dataset:

```bash pcqv1.sh```

To train Graphormer-Base on PCQM4Mv2 dataset:

```bash pcqv2.sh```


Note that the ```--batch-size``` should be modified accordingly to set the total batchsize as 1024.

#### PCQM4Mv2
Method        | #params | train MAE | valid MAE |
--------------|---------|-----------|-----------|
GCN          | 2.0M    | --    | 0.1379    |
GIN          | 3.8M    | --    | 0.1195    |
GCN-VN          | 4.9M    | --    | 0.1153    |
GIN-VN          | 6.7M    | --    | 0.1083    |
Graphormer-v2   | 47.1M   | 0.0253    | **0.0865**    |

#### PCQM4Mv1
Method        | #params | train MAE | valid MAE |
--------------|---------|-----------|-----------|
GCN          | 2.0M    | 0.1318    | 0.1691    |
GIN          | 3.8M    | 0.1203    | 0.1537    |
GCN-VN          | 4.9M    | 0.1225    | 0.1485    |
GIN-VN          | 6.7M    | 0.1150    | 0.1395    |
Graphormer-Small| 12.5M   | 0.0778    | 0.1264    |
Graphormer   | 47.1M   | 0.0582    | 0.1234    |
Graphormer-v2   | 47.1M   | 0.0309    | **0.1201**    |

## Open Graph Benchmark

The detailed description of the dataset could be found [here](https://ogb.stanford.edu/).

### Example Usage

Fine-tuning the pre-trained model on OGBG-MolHIV:

```
bash hiv_pre.sh
```

#### OGBG-MolHIV
Method        | #params | test AUC (%)|
--------------|---------|------------|
GCN-GraphNorm          | 526K    | 78.83      |
PNA          | 326K    | 79.05      |
PHC-GNN          | 111K    | 79.34      |
DeeperGCN-FLAG          | 532K    | 79.42      |
DGN          | 114K    | 79.70      |
Graphormer   | 47.0M   | 80.51      |
Graphormer-v2   | 47.1M   | **81.28**      |

## Benchmarking Graph Neural Networks - ZINC-500K


The detailed description of the dataset could be found [here](https://github.com/graphdeeplearning/benchmarking-gnns).

### Example Usage

To train Graphormer-Slim on ZINC-500K dataset:

```bash zinc.sh```

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



## Citation
Please kindly cite this paper if you use the code:
```
@inproceedings{
ying2021do,
title={Do Transformers Really Perform Badly for Graph Representation?},
author={Chengxuan Ying and Tianle Cai and Shengjie Luo and Shuxin Zheng and Guolin Ke and Di He and Yanming Shen and Tie-Yan Liu},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=OeWooOxFwDa}
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
