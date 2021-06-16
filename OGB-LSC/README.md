

## Graphormer

### Setup with Conda
```
# create a new environment
conda create --name graphormer-lsc python=3.7
conda activate graphormer-lsc
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric==1.6.3 ogb==1.3.1 pytorch-lightning==1.3.1 tqdm torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
conda install -c rdkit rdkit cython
```

### Prepare Dataset
```
cd OGB-LSC/graphormer/
python src/pcq_wrapper.py
```

### Train on PCQM4M
```
conda activate graphormer-lsc
export exp_name="all_fold"
export arch="--ffn_dim 768 --hidden_dim 768 --attention_dropout_rate 0.1 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20 --weight_decay 0.0 --intput_dropout_rate 0.0 --warmup_updates 10000 --tot_updates 1500000"
export seed="0"          # [0, 1]
export batch_size="256"
bash train.sh
```

### Download Checkpoints

Download GraphFormer pretrained model from [here](https://szheng.blob.core.windows.net/ogb-lsc/graphformer_checkpoints.tar.gz).

### Inference on Test Molecules
```
conda activate graphormer-lsc
export arch="--ffn_dim 768 --hidden_dim 768 --attention_dropout_rate 0.1 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20 --weight_decay 0.0 --intput_dropout_rate 0.0"
export ckpt_path="checkpoints"
checkpoint_names="all_fold_seed0 all_fold_seed1 fold0 fold1 fold2 fold3 fold4 fold5 fold6 fold7"
for val in $checkpoint_names; do
    export ckpt_name="$val.ckpt"
    bash inference.sh
done
```

## ExpC*

### Setup with Conda
```
# create a new environment
conda create --name expc-lsc python=3.6.9
conda activate expc-lsc
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge easydict
conda install -c conda-forge tensorboard
conda install -c rdkit rdkit

pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric==1.6.3
pip install ogb==1.3.1
```

### Prepare Dataset
When running train.sh or inference_expc_all.sh, the dataset would be downloaded automatically if it does not exist.

### Train on PCQM4M
```
conda activate expc-lsc
bash train.sh
```

### Download Checkpoints

Download ExpC* pretrained model from [here](https://szheng.blob.core.windows.net/ogb-lsc/expc_checkpoint_fold_0_7.zip).

### Inference on Test Molecules
```
conda activate expc-lsc
bash inference_expc_all.sh
```

## Final Ensemble
```
python ensemble.py
```
