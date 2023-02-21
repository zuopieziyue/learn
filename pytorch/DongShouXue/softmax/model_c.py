# -*- coding: utf-8 -*-


import torch
from torch import nn
from d2l import torch as d2l
from model import train_ch3


"""
PyTorch不会隐式地调整输入的形状
因此，在线性层前定义了展平层（flatten），来调整网络输入的形状
"""
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    """初始化权重参数"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)


"""损失函数"""
loss = nn.CrossEntropyLoss(reduction='none')


"""优化算法"""
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


if __name__ == '__main__':
    """主函数"""
    batch_size = 256
    num_epochs = 10

    # 读取数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 训练
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)






