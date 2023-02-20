# -*- coding: utf-8 -*-


import torch
from matplotlib import pyplot as plt


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    x = torch.normal(mean=0.0, std=1.0, size=(num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return x, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def show_figure():
    """绘制散点图"""
    plt.figure(figsize=(3.5, 2.5))
    plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy())
    plt.show()


if __name__ == '__main__':
    print(features[0])
    print(labels[0])
    show_figure()





