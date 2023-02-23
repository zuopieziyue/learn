# -*- coding: utf-8 -*-


import torch
from torch import nn
from d2l import torch as d2l
from train_ch3 import train_ch3


"""超参数"""
batch_size = 256
num_inputs = 784
num_hiddens = 256
num_outputs = 10
num_epochs = 10
lr = 0.1


def relu(X):
    """激活函数"""
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    """定义模型"""
    X = X.reshape((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    O = torch.matmul(H, W2) + b2
    return O


if __name__ == '__main__':
    """主函数"""

    # 初始化模型参数
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    # 获取数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 优化器
    updater = torch.optim.SGD(params, lr=lr)

    # 训练
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
