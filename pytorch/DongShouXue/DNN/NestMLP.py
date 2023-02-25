# -*- coding: utf-8 -*-


import torch
from torch import nn
from FixedHiddenMLP import FixedHiddenMLP
from torch.nn import functional as F


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


if __name__ == '__main__':
    """主函数"""
    X = torch.rand(2, 20)
    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    out = chimera(X)
    print(out)



