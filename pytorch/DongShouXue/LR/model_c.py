# -*- coding: utf-8 -*-


import torch
from torch import nn
from gen_dataset import features, labels
from dataset_c import data_iter


net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


loss = nn.MSELoss()


trainer = torch.optim.SGD(net.parameters(), lr=0.03)


num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print('epoch %d, loss %f' % (epoch + 1, l.mean()))











