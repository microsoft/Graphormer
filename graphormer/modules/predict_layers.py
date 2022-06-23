import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, activation=None, sandwich_norm=False, n_layers=1
    ):  
        super().__init__()
        assert sandwich_norm == False, "sandwich norm not supported"
        self.activation = activation

        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, in_dim, bias=True) for _ in range(n_layers - 1)]
        )
        self.fc_out = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        x = self.fc_out(x)
        return x


class PredictLayerGroup(nn.Module):
    def __init__(self, in_dim, out_dims, activation=None, sandwich_norm=False, n_layers=1):
        super().__init__()
        self.layers_list = nn.ModuleList(
            [PredictLayer(in_dim, out_dim, activation, sandwich_norm, n_layers) for out_dim in out_dims]
        )

    def forward(self, x):
        return [layer(x) for layer in self.layers_list]
