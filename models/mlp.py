import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, hidden_channels:int,
                 num_layers:int, dropout:float=0.5, batchnorm=False, bias=True,
                 res_connect=False) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.batchnorm = batchnorm
        self.res_connect = res_connect
        self.dropout = dropout
        layers = []
        if num_layers == 1: hidden_channels = in_channels
        assert num_layers > 0

        if batchnorm:
            batchnorm_layers = []
        else:
            batchnorm_layers = None

        for i in range(num_layers - 1):
            c_in = in_channels if i == 0 else hidden_channels
            c_out = hidden_channels
            layers.append(torch.nn.Linear(c_in, c_out, bias=bias))
            if batchnorm:
                batchnorm_layers.append(torch.nn.BatchNorm1d(c_out))
        layers.append(torch.nn.Linear(hidden_channels, out_channels, bias=bias))
        self.layers = torch.nn.ModuleList(layers)
        if batchnorm:
            self.batchnorm_layers = torch.nn.ModuleList(batchnorm)
        else:
            self.batchnorm_layers = None
    
    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[0](x)
        if self.num_layers == 1:
            return x
        if self.batchnorm:
            x = self.batchnorm_layers[0](x)
        x = F.relu(x)
        for i in range(1, self.num_layers - 1, 1):
            res_x = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[i](x)
            if self.batchnorm:
                x = self.batchnorm_layers[i](x)
            if self.res_connect:
                x = x + res_x
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)
