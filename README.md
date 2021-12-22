<img src="docs/logo-10.png" width=100%> 

Graphormer is a deep learning package that allows researchers and developers to train custom models for molecule modeling tasks. It aims to accelerate the research and application in AI for molecule science, such as material discovery, drug discovery, etc. [Project website](https://www.microsoft.com/en-us/research/project/graphormer/).

## Highlights in Graphormer v2.0
* The model, code, and script used in the [Open Catalyst Challenge](https://opencatalystproject.org/challenge.html) are available.
* Pre-trained models on PCQM4M and PCQM4Mv2 are available, more pre-trained models are comming soon.
* Supports interface and datasets of PyG, DGL, OGB, and OCP.
* Supports fairseq backbone.
* Document is online!

## What's New:

***12/22/2021***
  1. Graphormer v2.0 is released. Enjoy!

***12/10/2021***
  1. **Graphormer** has won the [Open Catalyst Challenge](https://opencatalystproject.org/challenge.html). The technical talk could be found through this [link](https://www.youtube.com/watch?v=uKJX3Mpu3OA&ab_channel=OpenCatalystProject).
  2. The **slides** of NeurIPS 2021 could be found through this [link](https://neurips.cc/media/neurips-2021/Slides/27679_XmkCZkZ.pdf).
  3. The **new release** of Graphormer is comming soon, as a general molecule modeling toolkit, with models used in OC dataset, completed pre-trained model zoo, flexible data interface, and higher effiency of training.

***09/30/2021***
  1. **Graphormer** has been accepted by **NeurIPS 2021**.
  2. We're hiring! Please contact ``shuz[at]microsoft.com`` for more information.

***08/03/2021***
  1. Codes and scripts are released.
  
***06/16/2021***
  1. Graphormer has won the **1st place** of quantum prediction track of Open Graph Benchmark Large-Scale Challenge (KDD CUP 2021) [[Competition Description]](https://ogb.stanford.edu/kddcup2021/pcqm4m/) [[Competition Result]](https://ogb.stanford.edu/kddcup2021/results/) [[Technical Report]](https://arxiv.org/pdf/2106.08279.pdf)   [[Blog (English)]](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/articles/transformer-stands-out-as-the-best-graph-learner-researchers-from-microsoft-research-asia-wins-the-kdd-cups-2021-graph-prediction-track/) [[Blog (Chinese)]](https://www.msra.cn/zh-cn/news/features/ogb-lsc) 

## Hiring
We are hiring at all levels (including FTE researchers and interns)! If you are interested in working with us on AI for Molecule Science, please send your resume to <a href="mailto:shuz@microsoft.com" class="x-hidden-focus">shuz@microsoft.com</a>.

## Get Started

Our primary documentation is at https://graphormer.readthedocs.io/ and is generated from this repository, which contains instructions for getting started, training new models and extending Graphormer with new model types and tasks.

Next you may want to read:

* [Examples](https://github.com/microsoft/Graphormer/tree/main/examples) showing command line usage of common tasks.


## Requirements and Installation

#### Setup with Conda

```
bash install.sh
```



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
