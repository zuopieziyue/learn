# -*- coding: utf-8 -*-


import random
import torch
from gen_dataset import features, labels


batch_size = 10


def data_iter(batch_size, features, labels):
    """
    遍历数据集，并不断读取小批量数据样本
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]  # take函数根据索引返回对应元素


if __name__ == '__main__':
    """主函数"""
    for X, y in data_iter(batch_size, features, labels):
        print(X)
        print(y)
        break