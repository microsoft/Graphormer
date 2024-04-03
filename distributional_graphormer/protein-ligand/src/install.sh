#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# install requirements
pip install torch==1.9.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb==1.3.0
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-geometric==2.0.4
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html

python setup_cython.py build_ext --inplace

# if fairseq submodule has not been checkouted, run:
git clone https://github.com/pytorch/fairseq
cd fairseq
git reset --hard 98ebe4f1ada75d00
pip install -e .
python setup.py build_ext --inplace

cd ..
# pre-compiling cython code
python setup_cython.py build_ext --inplace

# conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
