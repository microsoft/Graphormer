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
yes | pip install torch-geometric==1.7.2
yes | pip install tensorboardX==2.4.1
yes | pip install ogb==1.3.2
yes | pip install rdkit-pypi==2021.9.3
yes | pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html

git submodule update --init --recursive
cd fairseq
git reset --hard 98ebe4f1ada75d006717d84f9d603519d8ff5579
yes | pip install .
python setup.py build_ext --inplace

yes | pip install icecream
yes | pip install torchmetrics
yes | pip install PyTDC
conda install -c conda-forge tensorboard -y
yes | pip install omegaconf
yes | pip install Cython
cd ..
python setup_cython.py build_ext --inplace
yes | pip install numpy==1.20.0
