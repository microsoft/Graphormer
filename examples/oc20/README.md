# Open Catalyst Challenge


<img src="ocp.gif" width=70%> 

The Open Catalyst Project is a collaborative research effort between Facebook AI Research (FAIR) and Carnegie Mellon University’s (CMU) Department of Chemical Engineering. The aim is to use AI to model and discover new catalysts for use in renewable energy storage to help in addressing climate change.

The detailed description of this dataset could be found in [here](https://opencatalystproject.org/).



### Example Usage

Data Preparation: Follow the instructions [here](https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET.md) to prepare your OC20 dataset.

To train a Graphormer-3D for IS2RE direct task:

```bash oc20.sh```

#### IS2RE Direct Energy MAE (eV) on test split

Method        | ID      | OOD Ads   | OOD Cat   | OOD Both  |  Avg    |
--------------|---------|-----------|-----------|-----------|---------|
CGCNN         | 0.6149  | 0.9155    |  0.6219   | 0.8511    |  0.7509 | 
SchNet        | 0.6387  | 0.7342    |  0.6616   | 0.7037    |  0.6846 |
DimeNet++     | 0.5620  | 0.7252    | 0.5756    | 0.6613    |  0.6311 | 
SphereNet     | 0.5625  |  0.7033   |  0.5708   | 0.6378    | 0.6186  |
SpinConv      | 0.5583  | 0.7230    | 0.5687    | 0.6738    |  0.6310 |
Noisy Node    | 0.4776  | 0.5646    | 0.4932    | 0.5042    | 0.5099  |
Graphormer-3D (ensemble) | 0.3976  | 0.5719    | 0.4166    | 0.5029    |  0.4722   |  

*note: Evaluation of model performance on test split requires submission through [EvalAI](https://eval.ai/web/challenges/challenge-page/712/overview).


## Citation
Please kindly cite this paper if you use the code:
```
@article{shi2022benchmarking,
  title={Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets},
  author={Yu Shi and Shuxin Zheng and Guolin Ke and Yifei Shen and Jiacheng You and Jiyan He and Shengjie Luo and Chang Liu and Di He and Tie-Yan Liu},
  journal={arXiv preprint arXiv:2203.04810},
  year={2022},
  url={https://arxiv.org/abs/2203.04810}
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






