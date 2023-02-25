# -*- coding: utf-8 -*-


import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """自定义块"""
    # 用模型参数声明层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  #输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


if __name__ == '__main__':
    """主函数"""
    X = torch.rand(2, 20)
    net = MLP()
    out = net(X)
    print(out)