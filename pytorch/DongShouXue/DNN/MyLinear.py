# -*- coding: utf-8 -*-


import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    """自定义不带参数的层"""
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
out = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
print(out)


class MyLinear(nn.Module):
    """自定义带参数的层"""
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight)


net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 2))
print(net(torch.rand(2, 64)))

