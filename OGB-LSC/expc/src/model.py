# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from src.utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from src.conv import CombConv
from src.utils.mol_encoder import AtomEncoder


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_tasks=1):
        super(Net, self).__init__()

        self.layers = config.layers

        self.atom_encoder = AtomEncoder(config.hidden)

        self.virtualnode_embedding = torch.nn.Embedding(1, config.hidden)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.convs = torch.nn.ModuleList()
        for i in range(config.layers):
            self.convs.append(CombConv(config))

        self.JK = JumpingKnowledge(config.JK)

        if config.JK == 'C':
            num_linear = config.layers
        else:
            num_linear = 1

        self.graph_pred_linear = Sequential(
            Linear(config.hidden * num_linear, config.hidden * num_linear),
            ReLU(),
            Linear(config.hidden * num_linear, num_tasks))

        if config.pooling == 'S':
            self.pool = global_add_pool
        elif config.pooling == 'M':
            self.pool = global_mean_pool

        self.dropout = config.dropout

        self.virtualnode_mlp1 = torch.nn.ModuleList()
        self.virtualnode_mlp2 = torch.nn.ModuleList()
        for layer in range(config.layers):
            self.virtualnode_mlp1.append(torch.nn.Sequential(torch.nn.Linear(config.hidden * 2, config.hidden * 2),
                                                             torch.nn.BatchNorm1d(config.hidden * 2),
                                                             ReLU()))

            self.virtualnode_mlp2.append(torch.nn.Sequential(torch.nn.Linear(config.hidden * 2, config.hidden),
                                                             torch.nn.BatchNorm1d(config.hidden),
                                                             ReLU()))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        x = self.atom_encoder(x)
        vxs = []
        for i, conv in enumerate(self.convs):
            x = x + virtualnode_embedding[batch]
            x = conv(x, edge_index, edge_attr)

            virtualnode_embedding_temp = F.dropout(self.virtualnode_mlp1[i](torch.cat([virtualnode_embedding[batch], x], dim=-1)), self.dropout,
                                                   training=self.training)
            pooled_virtualnode_embedding = global_add_pool(virtualnode_embedding_temp, batch)
            virtualnode_embedding = F.dropout(self.virtualnode_mlp2[i](pooled_virtualnode_embedding), self.dropout,
                                              training=self.training)

            vxs += [virtualnode_embedding]

        vnr = self.JK(vxs)
        vnr = F.dropout(vnr, p=self.dropout, training=self.training)
        h_graph = vnr

        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference_tool time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)

    def __repr__(self):
        return self.__class__.__name__
