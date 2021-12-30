Start with Example
==================

Graphormer provides example scripts to train your own models on several datasets.
For example, to train a Graphormer-slim on ZINC-500K on a single GPU card:

.. code-block:: console

    > cd examples/property_prediction/
    > bash zinc.sh

The content of ``zinc.sh`` is simply a ``fairseq-train`` command:

.. code-block:: console

    CUDA_VISIBLE_DEVICES=0 fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name zinc \
        --dataset-source pyg \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_slim \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 400000 \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size 64 \
        --fp16 \
        --data-buffer-size 20 \
        --encoder-layers 12 \
        --encoder-embed-dim 80 \
        --encoder-ffn-embed-dim 80 \
        --encoder-attention-heads 8 \
        --max-epoch 10000 \
        --save-dir ./ckpts

``CUDA_VISIBLE_DEVICES`` specifies the GPUs to use. With multiple GPUs, the GPU IDs should be separated by commas.
A ``fairseq-train`` with Graphormer model is used to launch training.
:ref:`Command-line Tools` gives detailed explanations to the parameters.

Similarily, to train a Graphormer-base on PCQM4M dataset on multiple GPU cards:

.. code-block:: console

    > cd examples/property_prediction/
    > bash pcqv1.sh
    

By runing the instructions in the scripts,  Graphormer will automatically download the needed datasets and pre-process them.


Evaluate Pre-trained Models
===========================

Graphormer provides pretrained models so that users can easily evaluate, and finetune.
To evaluate a pre-trained model, use the script ``graphormer/evaluate/evaluate.py``.

.. code-block:: console

    python evaluate.py \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name pcqm4m \
        --dataset-source ogb \
        --task graph_prediction \
        --criterion l1_loss \
        --arch graphormer_base \
        --num-classes 1 \
        --batch-size 64 \
        --pretrained-model-name pcqm4mv1_graphormer_base \
        --load-pretrained-model-output-layer \
        --split valid \
        --seed 1

``--pretrained-model-name`` specifies the pre-trained model to be valuated. The pre-trained model will be automatically downloaded. And ``--load-pretrained-model-output-layer`` is set so that weights of the
final fully connected layer in the pre-trained model is loaded. And ``--split`` specifies the split of the dataset to be evaluated, can be ``train`` or ``valid``.

Fine-tuning Pre-trained Models
==============================
To fine-tune pre-trained models, use ``--pretrained-model-name`` to set the model name. For example, the script ``examples/property_prediction/hiv_pre.sh``
fine-tunes our model ``pcqm4mv1_graphormer_base`` on the ``ogbg-molhiv`` dataset. The command for fine-tune is

.. code-block:: console

    fairseq-train \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name ogbg-molhiv \
        --dataset-source ogb \
        --task graph_prediction_with_flag \
        --criterion binary_logloss_with_flag \
        --arch graphormer_base \
        --num-classes 1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --weight-decay 0.0 \
        --lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
        --lr 2e-4 --end-learning-rate 1e-9 \
        --batch-size $batch_size \
        --fp16 \
        --data-buffer-size 20 \
        --encoder-layers 12 \
        --encoder-embed-dim 768 \
        --encoder-ffn-embed-dim 768 \
        --encoder-attention-heads 32 \
        --max-epoch $max_epoch \
        --save-dir ./ckpts \
        --pretrained-model-name pcqm4mv1_graphormer_base \
        --flag-m 3 \
        --flag-step-size 0.001 \
        --flag-mag 0.001 \
        --seed 1

After fine-tuning, use ``graphormer/evaluate/evaluate.py`` to evaluate the performance of all checkpoints:

.. code-block:: python

    python evaluate.py \
        --user-dir ../../graphormer \
        --num-workers 16 \
        --ddp-backend=legacy_ddp \
        --dataset-name ogbg-molhiv \
        --dataset-source ogb \
        --task graph_prediction \
        --arch graphormer_base \
        --num-classes 1 \
        --batch-size 64 \
        --save-dir ../../examples/property_prediction/ckpts/ \
        --split test \
        --metric auc \
        --seed 1


Training a New Model
====================

We take OC20 as an example to show how to train a new model on your own datasets.

First, download IS2RE train, validation, and test data in LMDB format by:

.. code-block:: console

    > cd examples/oc20/ && mkdir data && cd data/
    > wget -c https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz && tar -xzvf is2res_train_val_test_lmdbs.tar.gz 
    
Create ``ckpt`` folder to save checkpoints during the training:

.. code-block:: console

    > cd ../ && mkdir ckpt/
    
Now we train a 48-layer ``graphormer-3D`` architecture, which has 4 blocks and each block contains 12 Graphormer layers. The parameters are sharing across blocks. The total training steps are 1 million, and we warmup the learning rate by 10 thousand steps. 

.. code-block:: console

    > fairseq-train --user-dir ../../graphormer  \
       ./data/is2res_train_val_test_lmdbs/data/is2re/all --valid-subset val_id,val_ood_ads,val_ood_cat,val_ood_both --best-checkpoint-metric loss \
       --num-workers 0 --ddp-backend=c10d \
       --task is2re --criterion mae_deltapos --arch graphormer3d_base  \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm $clip_norm \
       --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates  --total-num-update 1000000 --batch-size 4 \
       --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 1 --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir ./tsbs \
       --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 \
       --max-update 1000000 --log-interval 100 --log-format simple \
       --save-interval-updates 5000 --validate-interval-updates 2500 --keep-interval-updates 30 --no-epoch-checkpoints  \
       --save-dir ./ckpt --layers 12 --blocks 4 --required-batch-size-multiple 1  --node-loss-weight 15

Please note that ``--batch-size 4`` requires at least 32GB of GPU memory. If out of GPU momery occuars, one may try to reduce the batchsize then train with more GPU cards, or increase the ``--update-freq`` to accumulate the gradients.
