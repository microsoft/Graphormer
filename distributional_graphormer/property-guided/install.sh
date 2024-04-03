#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

pip uninstall -y setuptools
yes | pip install setuptools==59.5.0
# install requirements
yes | pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
yes | pip install lmdb
yes | pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
yes | pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
yes | pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
yes | pip install torch-geometric==1.7.2
yes | pip install tensorboardX==2.4.1
yes | pip install ogb==1.3.2
yes | pip install rdkit-pypi==2021.9.3
yes | pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
yes | pip install m3gnet
yes | pip install PyAstronomy

cd fairseq
# if fairseq submodule has not been checkouted, run:
# git submodule update --init --recursive
yes | pip install .
python setup.py build_ext --inplace

yes | pip install icecream
yes | pip install torchmetrics
yes | pip install PyTDC
conda install -c conda-forge tensorboard -y
yes | pip install omegaconf
yes | pip install wandb
wandb login 5e7c43fe99d4525ad4abd80ceac0e18d03193e31
cd ..
python setup_cython.py build_ext --inplace
