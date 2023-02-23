# -*- coding: utf-8 -*-


import torch
from torch import nn
from d2l import torch as d2l
from train_ch3 import train_ch3


"""超参数"""
num_inputs = 784
num_hiddens1 = 256
num_hiddens2 = 256
num_outputs = 10
dropout1 = 0.5
dropout2 = 0.5

num_epochs = 10
lr = 0.1
batch_size = 256


def dropout_layer(X, dropout):
    """dropout层的实现"""
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


class Net(nn.Module):
    """定义模型结构"""
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


if __name__ == '__main__':
    """主函数"""
    # 初始化模型参数
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    # 获取数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # loss函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练过程
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)











