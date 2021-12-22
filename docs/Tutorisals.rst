Simple MLP Tutorial
===================

In this tutorial, we will extend Graphormer by adding a new :class:`~graphormer.models.GraphMLP` that transforms the node features, and uses a sum pooling layer to combine the output of the MLP as graph representation.

This tutorial covers:

1. **Writing a new Model** so that the node token embeddings can be transformed by the MLP.
2. **Training the Model** using the existing command-line tools.

1. Writing a new GraphMLP Model
--------------------------------

First, we create a new file with filename :file:`graphormer/models/graphmlp.py`::

 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 from fairseq.models import FairseqEncoderModel, register_model 

 @register_model("graphmlp")
 class GraphMLP(FairseqEncoderModel):
     def __init__(self, args, encoder):
         super().__init__(encoder)
         self.args = args

     @staticmethod
     def add_args(parser):
         """Add model-specific arguments to the parser."""
         parser.add_argument(
             "--encoder-layers", type=int, metavar="N", help="num encoder layers"
         )
         parser.add_argument(
             "--max-nodes", type=int, metavar="N", help="num max nodes"
         )
         parser.add_argument(
             "--encoder-embed-dim",type=int, metavar="N", help="encoder embedding dimension",
         )

     def max_nodes(self):
         return self.encoder.max_nodes

     @classmethod
     def build_model(cls, args, task):
         """Build a new model instance."""
         # make sure all arguments are present in older models
         graphmlp_architecture(args)
         encoder = GraphMLPEncoder(args)
         return cls(args, encoder)
 
     def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)

The main component in :class:`~graphormer.models.GraphMLP` is the :class:`~graphormer.models.GraphMLPEncoder`. Here we implement it by adding following codes in :file:`graphormer/models/graphmlp.py`::

 from fairseq.models import FairseqEncoder
 from ..modules import GraphNodeFeature

 class GraphMLPEncoder(FairseqEncoder):
     def __init__(self, args):
         super().__init__(dictionary=None)
         self.max_nodes = args.max_nodes
         self.emb_dim = args.encoder_embed_dim
         self.num_layer = args.encoder_layers
         self.num_classes = args.num_classes
 
         self.atom_encoder = GraphNodeFeature(
             num_heads=1,
             num_atoms=512*9,
             num_in_degree=512,
             num_out_degree=512,
             hidden_dim=self.emb_dim,
             n_layers=self.num_layer,
         )

         self.linear = torch.nn.ModuleList()
         self.batch_norms = torch.nn.ModuleList()

         for layer in range(self.num_layer):
             self.linear.append(torch.nn.Linear(self.emb_dim, self.emb_dim))
             self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))
         

         self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_classes)


     def forward(self, batched_data, **unused):
         h=self.atom_encoder(batched_data)
         for layer in range(self.num_layer):
             h = self.linear[layer](h)
             h = h.transpose(1,2)
             h = self.batch_norms[layer](h)
             h = h.transpose(1,2)

             if layer != self.num_layer - 1:
                 h = F.relu(h)
     
         h = h.sum(dim=1)
         out = self.graph_pred_linear(h)

         return out.unsqueeze(1)


     def max_nodes(self):
         return self.max_nodes

Since we will validate our GraphMLP model on a graph representation task, we choose dataset in MoleculeNet. Therefore, we employ the :class:`~graphormer.modules.GraphNodeFeature` to encode the node features.

And finally, we register the model architecture by adding following codes in :file:`graphormer/models/graphmlp.py`::

 from fairseq.models import register_model_architecture
 @register_model_architecture("graphmlp", "graphmlp")
 def graphmlp_architecture(args):
     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
     args.encoder_layers = getattr(args, "encoder_layers", 12)

     args.max_nodes = getattr(args, "max_nodes", 512)

2. Training the Model
---------------------

Next, we prepare the training script for the model. We create a bash file :file:`examples/property_prediction/graphmlp.sh`::

 #!/bin/bash
 CUDA_VISIBLE_DEVICES=0 fairseq-train \
 --user-dir ../../graphormer \
 --num-workers 16 \
 --ddp-backend=legacy_ddp \
 --dataset-name moleculenet:name=bbbp   \
 --dataset-source pyg \
 --task graph_prediction \
 --criterion binary_logloss \
 --arch graphmlp \
 --num-classes 1 \
 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
 --lr-scheduler polynomial_decay --power 1 --total-num-update 1000000 \
 --lr 0.001 --end-learning-rate 1e-9 \
 --batch-size 32 \
 --fp16 \
 --data-buffer-size 20 \
 --encoder-layers 5 \
 --encoder-embed-dim 256 \
 --max-epoch 100 \
 --save-dir ./ckpts \
 --save-interval-updates 50000 \
 --no-epoch-checkpoints

By executing the script, after the dataset is downloaded and processed, the training of the GraphMLP model starts.
