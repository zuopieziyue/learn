# -*- coding: utf-8 -*-


import torch
from gen_dataset import features, labels
from dataset import batch_size, data_iter


w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    """模型参数"""
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    """模型训练过程"""
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()  # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使用小批量梯度下降迭代模型参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print('epoch %d, loss %f' % (epoch + 1, train_l.mean()))






