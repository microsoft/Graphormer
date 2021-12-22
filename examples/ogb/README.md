# Open Graph Benchmark

[https://arxiv.org/abs/2005.00687](https://arxiv.org/abs/2005.00687)


[https://ogb.stanford.edu/](https://ogb.stanford.edu/)

## Results

#### OGBG-MolPCBA
Method        | #params | test AP (%)|
--------------|---------|------------|
DeeperGCN-VN+FLAG         | 5.6M    | 28.42      |
DGN          | 6.7M    | 28.85      |
GINE-VN          | 6.1M    | 29.17      |
PHC-GNN          | 1.7M    | 29.47      |
GINE-APPNP          | 6.1M    | 29.79      |
Graphormer   | 119.5M  | **31.39**      |

#### OGBG-MolHIV
Method        | #params | test AP (%)|
--------------|---------|------------|
GCN-GraphNorm          | 526K    | 78.83      |
PNA          | 326K    | 79.05      |
PHC-GNN          | 111K    | 79.34      |
DeeperGCN-FLAG          | 532K    | 79.42      |
DGN          | 114K    | 79.70      |
Graphormer   | 47.0M   | **80.51**      |

## Example Usage

Prepare your pre-trained models following our paper ["Do Transformers Really Perform Bad for Graph Representation?"](https://arxiv.org/abs/2106.05234).

Fine-tuning your pre-trained model on OGBG-MolPCBA:

```
bash pcba.sh
```

Fine-tuning your pre-trained model on OGBG-MolHIV:

```
bash hiv.sh
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
