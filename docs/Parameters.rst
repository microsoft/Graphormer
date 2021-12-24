.. _Command-line Tools:

Command-line Tools
==================

Graphormer reuses the ``fairseq-train`` command-line tools of `fairseq <https://fairseq.readthedocs.io/en/latest/command_line_tools.html>`__ for training, and here we mainly document the additional parameters in Graphormer
and parameters of ``fairseq-train`` used by Graphormer.

Model
-----
-  ``--arch``, type=enum, options: ``graphormer_base``, ``graphormer_slim``, ``graphormer_large``
   
   -  Predefined graphormer architectures

-  ``--encoder-ffn-embed-dim``, type = float

   -  encoder embedding dimension for FFN

-  ``--encoder-layers``, type = int

   -  number of graphormer encoder layers

-  ``--encoder-embed-dim``, type = int

   -  encoder embedding dimension

-  ``--share-encoder-input-output-embed``, type = bool

   -  if set, share encoder input and output embeddings

-  ``--share-encoder-input-output-embed``, type = bool

   -  if set, share encoder input and output embeddings

-  ``--encoder-learned-pos``, type = bool

   -  if set, use learned positional embeddings in the encoder

-  ``--no-token-positional-embeddings``, type = bool

   -  if set, disables positional embeddings" " (outside self attention)

-  ``--max-positions``, type = int

   -  number of positional embeddings to learn

-  ``--activation-fn``, type = enum, options: ``gelu``, ``relu``

   -  activation function to use

-  ``--encoder-normalize-before``

   -  if set, apply layernorm before each encoder block


Training
--------
-  ``--task``, type = enum, options: ``graph_prediction``, ``is2re``

   -  the task for training

      - ``graph_prediction``: ordinary graph-level prediction task, predict a single target for a graph

      - ``is2re``: for IS2RE task of `Open Catalyst Challenge`_

-  ``--criterion``, type = enum, options: ``l1_loss``, ``binary_logloss``, ``multiclass_cross_entropy``, ``mae_deltapos``, ``l1_loss_with_flag``, ``binary_logloss_with_flag``, ``multiclass_cross_entropy_with_flag``

   -  the criterion, or objective function for training.

      - ``l1_loss``: mean absolute error (MAE) for regression tasks

      - ``binary_logloss``: binary cross entropy for binary classification

      - ``multiclass_cross_entropy``: multi-class cross entropy for multi-class classification

      - ``mae_deltapos``: criterion for IS2RE task of `Open Catalyst Challenge`_

      - ``l1_loss_with_flag``: ``l1_loss`` with FLAG_

      - ``binary_logloss_with_flag``: ``binary_logloss`` with FLAG_

      - ``multiclass_cross_entropy_with_flag``: ``multiclass_cross_entropy`` with FLAG_

-  ``--apply-graphormer-init``, type = bool

   -  if set, use custom param initialization for Graphormer

-  ``--dropout``, type = float

   -  dropout probability

-  ``--attention-dropout``, type = float

   -  dropout probability for attention weights

-  ``--act-dropout``, type = float

   -  dropout probability after activation in FFN

-  ``--seed``, type = int

   -  random seed

-  ``--pretrained-model-name``, type = enum, default= ``none``, options: ``pcqm4mv1_graphormer_base``, ``pcqm4mv2_graphormer_base``

   -  name of used pretrained model

      - ``pcqm4mv1_graphormer_base``: Pretrained Graphormer base model with `PCQM4M v1 <https://ogb.stanford.edu/kddcup2021/pcqm4m/>`__ dataset.

      - ``pcqm4mv2_graphormer_base``: Pretrained Graphormer base model with `PCQM4M v2 <https://ogb.stanford.edu/docs/lsc/pcqm4mv2/>`__ dataset.

-  ``--load-pretrained-model-output-layer``, type = bool

   -  if set, the weights of the final fully connected layer in the pre-trained model is loaded

-  ``--optimizer``, type = enum

   -  optimizers from `fairseq <https://fairseq.readthedocs.io/en/latest/optim.html>`__

-  ``--lr``, type = float

   -  learning rate

-  ``--lr-scheduler``, type=enum

   - learning rate scheduler from `fairseq <https://fairseq.readthedocs.io/en/latest/lr_scheduler.html>`__

-  ``--fp16``, type=bool

   - if set, use mixed precision training

-  ``--data-buffer-size``, type=int, default=10

   - number of batches to preload

-  ``--batch-size``, type=int

   - number of examples in a batch

-  ``--max-epoch``, type=int, default=0

   - force stop training at specified epoch

-  ``--save-dir``, type=str, default=``checkpoints``

   - path to save checkpoints


Dataset
-------
-  ``--dataset-name``, type = str, default= ``pcqm4m``

   -  name of the dataset

-  ``--dataset-source``, type = str, default= ``ogb``

   -  source of graph dataset, can be: ``pyg``, ``dgl``, ``ogb``

-  ``--num-classes``, type = int, default=-1

   -  number of classes or regression targets

-  ``--num-atoms``, type = int, default=512 * 9

   -  number of atom types in the graph

-  ``--num-edges``, type = int, default=512 * 3

   -  number of edge types in the graph

-  ``--num-in-degree``, type = int, default=512

   -  number of in degree types in the graph

-  ``--num-out-degree``, type = int, default=512

   -  number of out degree types in the graph

-  ``--num-spatial``, type = int, default=512

   -  number of spatial types in the graph

-  ``--num-edge-dis``, type = int, default=128

   -  number of edge dis types in the graph

-  ``--multi-hop-max-dist``, type = int, default=5

   -  max number of edges considered in the edge encoding

-  ``--spatial-pos-max``, type = int, default=1024

   -  max distance of attention in graph

-  ``--edge-type``, type = str, default="multi_hop"

   -  edge type in the graph

-  ``--edge-type``, type = str, default="multi_hop"

   -  edge type in the graph

-  ``--user-data-dir``, type = str, default=""

   -  path to the module of user-defined dataset


.. _Open Catalyst Challenge: https://opencatalystproject.org/challenge.html
.. _FLAG: https://arxiv.org/abs/2010.09891