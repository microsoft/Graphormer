# Graphormer

By [Chengxuan Ying](https://github.com/chengxuanying/), [Tianle Cai](https://tianle.website/), [Shengjie Luo](https://github.com/lsj2408), [Shuxin Zheng](https://www.microsoft.com/en-us/research/people/shuz/)\*, [Guolin Ke](https://github.com/guolinke), [Di He](https://www.microsoft.com/en-us/research/people/dihe/)\*, [Yanming Shen](https://dblp.org/pid/51/3800.html) and [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/).

This repo is the official implementation of ["Do Transformers Really Perform Bad for Graph Representation?"](https://arxiv.org/abs/2106.05234). 

## Updates

***06/10/2021***

Initial commits:

1. License files and example code.


## Introduction

**Graphormer** is initially described in [arxiv](https://arxiv.org/abs/2106.05234), which is a standard Transformer architecture with several structural encodings, which could effectively encoding the structural information of a graph into the model. 

Graphormer achieves strong performance on PCQM4M-LSC (`0.1234 MAE` on val), MolPCBA (`31.39 AP(%)` on test), MolHIV (`80.51 AUC(%)` on test) and ZINC (`0.122 MAE on test`), surpassing previous models by a large margin.



## Main Results 


## Citing Graphormer

```
@article{ying2021transformers,
  title={Do Transformers Really Perform Bad for Graph Representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2106.05234},
  year={2021}
}
```

## Getting Started


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
