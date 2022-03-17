#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# install requirements
pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-geometric==1.7.2
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html

cd fairseq
# if fairseq submodule has not been checkouted, run:
# git submodule update --init --recursive
pip install . --use-feature=in-tree-build
python setup.py build_ext --inplace
