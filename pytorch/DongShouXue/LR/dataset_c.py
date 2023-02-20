# -*- coding: utf-8 -*-


import random
import torch
from torch.utils import data
from gen_dataset import features, labels


batch_size = 10


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))


if __name__ == '__main__':
    """主函数"""
    print(data_iter)


