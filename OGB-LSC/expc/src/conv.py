# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops
from torch.nn import LeakyReLU, Linear, Sequential, Tanh, ELU, GELU, ReLU, BatchNorm1d as BN
from src.utils.mol_encoder import BondEncoder


class CombConv(MessagePassing):
    def __init__(self, config, **kwargs):
        super(CombConv, self).__init__(aggr='add', **kwargs)

        if config.exp_nonlinear == 'Tanh':
            self.exp_nonlinear = Tanh()
        elif config.exp_nonlinear == 'ELU':
            self.exp_nonlinear = ELU()
        elif config.exp_nonlinear == 'GELU':
            self.exp_nonlinear = GELU()
        elif config.exp_nonlinear[:6] == 'LeakyR':
            self.exp_nonlinear = LeakyReLU(float(config.exp_nonlinear[6:]))
        else:
            raise ValueError('Wrong exp_nonlinear called {}'.format(config.exp_nonlinear))

        self.exp_node = Sequential(
            Linear(config.hidden, config.hidden * config.exp_n),
            self.exp_nonlinear)
        self.exp_edge = Sequential(
            Linear(config.hidden, config.hidden * config.exp_n),
            self.exp_nonlinear)

        if config.exp_bn == 'Y':
            self.exp_bn = BN(config.hidden * config.exp_n)
        else:
            self.exp_bn = None

        self.fea_mlp = Sequential(
            Linear(config.hidden * config.exp_n, config.hidden * config.exp_n),
            ReLU(),
            Linear(config.hidden * config.exp_n, config.hidden),
            ReLU(),
            BN(config.hidden))

        self.bond_encoder = BondEncoder(emb_dim=config.hidden)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.bond_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.exp_bn is not None:
            x = self.exp_bn(x)

        x = self.fea_mlp(x)

        return x

    def message(self, x_i, x_j, edge_attr):
        return self.exp_edge(edge_attr) * self.exp_node(x_j)

    def update(self, aggr_out, x):
        return aggr_out + self.exp_node(x)

    def __repr__(self):
        return self.__class__.__name__

