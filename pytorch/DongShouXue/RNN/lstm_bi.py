from torch import nn
from n_gram import load_data_time_machine
from rnn import train_ch8
from rnn_c import RNNModel
from d2l import torch as d2l


"""获取数据集"""
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


"""双向LSTM模型"""
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
"""通过设置bidirective=True来定义双向"""
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)


"""训练模型"""
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, device)


