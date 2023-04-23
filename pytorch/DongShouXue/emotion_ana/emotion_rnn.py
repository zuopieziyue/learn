import torch
from torch import nn
from d2l import torch as d2l


print("""读取数据集""")
batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)


class BiRNN(nn.Module):
    """双向循环神经网络，情感分析"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


print("""初始化网络参数""")
embed_size, num_hiddens, num_layers = 100, 100, 2
devices = d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)


def init_weights(m):
    """初始化权重参数"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if 'weight' in param:
                nn.init.xavier_uniform_(m._parameters[param])


net.apply(init_weights)


print("""加载预训练的100维的GloVe嵌入""")
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')


print("""打印词表中所有词元的形状""")
embeds = glove_embedding[vocab.idx_to_token]
print(embeds.shape)


print("""使用预训练的词向量来表示评论中的词元，并且在训练中不要更新这些向量""")
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False


print("""训练双向循环神经网络进行情感分析""")
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


def predict_sentiment(net, vocab, sequence):
    """预测文本序列的情感"""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


print("""情感分类测试""")
predict_sentiment(net, vocab, 'this movie is so bad')



