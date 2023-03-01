# -*- coding: utf-8 -*-


import torch
from torch import nn


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
out = net(X)
print(out)


"""获取参数"""
print(net[2].state_dict())


def init_normal(m):
    """初始化为标准差为0.01的高斯随机变量，将偏置参数设置为0"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])


def init_constant(m):
    """初始化为常数"""
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])


def init_xavier(m):
    """Xavier初始化方法"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


net[0].apply(init_xavier)
print(net[0].weight.data[0], net[0].bias.data[0])

