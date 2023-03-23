import torch
from torch import nn
from n_gram import load_data_time_machine
from rnn import train_ch8
from rnn_c import RNNModel
from d2l import torch as d2l


"""获取数据集"""
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


"""深度循环神经网络简洁实现"""
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)


"""训练和预测"""
num_epochs, lr = 10, 2
train_ch8(model, train_iter, vocab, lr * 1.0, num_epochs, device)








