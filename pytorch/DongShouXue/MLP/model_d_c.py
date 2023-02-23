# -*- coding: utf-8 -*-


import torch
from torch import nn
from d2l import torch as d2l
from train_ch3 import train_ch3


"""超参数"""
dropout1 = 0.5
dropout2 = 0.5

num_epochs = 10
lr = 0.1
batch_size = 256


net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    # 在第一个全连接层之后添加一个dropout层
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    # 在第二个全连接层之后添加一个dropout层
    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    """主函数"""
    # 初始化模型参数
    net.apply(init_weights)

    # 获取数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # loss函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练过程
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)











