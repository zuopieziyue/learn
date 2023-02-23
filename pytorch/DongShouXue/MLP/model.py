# -*- coding: utf-8 -*-


import torch
from torch import nn
from d2l import torch as d2l


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


loss = nn.CrossEntropyLoss(reduction='none')


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    """
    对多个变量进行累加。
    存储两个变量，分别用于存储正确预测的数量和预测的总数量。
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内部的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型。多个迭代周期，每个迭代周期结束时，利用test_iter访问到的测试数据集对模型进行评估"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print('epoch: ', epoch,
              'train_loss: ', train_metrics[0],
              'train_acc: ', train_metrics[1],
              'test_acc: ', test_acc)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


if __name__ == '__main__':
    """主函数"""
    # 超参数
    batch_size = 256
    num_inputs = 784
    num_hiddens = 256
    num_outputs = 10
    num_epochs = 10
    lr = 0.1

    # 初始化模型参数
    """初始化模型参数"""
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    # 获取数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 训练
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
