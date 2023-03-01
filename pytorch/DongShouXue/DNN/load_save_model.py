# -*- coding: utf-8 -*-


import torch
from torch import nn
from torch.nn import functional as F


"""保存张量"""
x = torch.arange(4)
torch.save(x, 'model_file/x-file')


x2 = torch.load('model_file/x-file')
print(x2)


class MLP(nn.Module):
    """加载和保存模型参数"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), "model_file/mlp.param")


clone = MLP()
clone.load_state_dict(torch.load('model_file/mlp.param'))
print(clone.eval())









