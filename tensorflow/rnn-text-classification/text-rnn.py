# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:40:56 2019

@author: gongyue
"""
# ==============================================================================
# 构建计算图--LSTM模型
#    embedding
#    LSTM
#    fc
#    train_op
# 训练流程代码
# 数据集封装
#    next_batch(batch_size)
# 词表封装：
#    api: sentence2id(text_sentence): 句子转换id
# 类别的封装：
#    api: category2id(text_category): 类别转换id
# ==============================================================================

import sys
import os
import math
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# 定义参数，使用tensorflow的api来实现，便于管理
def get_default_params():
	return tf.contrib.training.HParams(
		num_embedding_size = 16,
		num_timesteps = 50,
		num_lstm_nodes = [32, 32],
		num_lstm_layers = 2,
		num_fc_nodes = 32,
		batch_size = 100,
		clip_lstm_grads = 1.0,
		learning_rate = 0.001,
		num_word_threshold = 10
	)

hps = get_default_params()


# 文件路径定义
train_file = 'text_classification_data\\cnews.train.seg.txt'
val_file = 'text_classification_data\\cnews.val.seg.txt'
test_file = 'text_classification_data\\cnews.test.seg.txt'
vocab_file = 'text_classification_data\\cnews.vocab.txt'
category_file = 'text_classification_data\\cnews.category.txt'
output_folder = 'text_classification_data\\run_text_rnn'

if not os.path.exists(output_folder):
	os.mkdir(output_folder)


# 定义词表封装
class Vocab:
	def __init__(self, filename, num_word_threshold):
		self._word_to_id = {}
		self._unk = -1
		self._num_word_threshold = num_word_threshold
		self._read_dict(filename)
	
	def _read_dict(self, filename):
		with open(filename, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		for line in lines:
			word, frequency = line.strip('\n').split('\t')
			frequency = int(frequency)
			if frequency < self._num_word_threshold:
				continue
			idx = len(self._word_to_id)
			if word == '<UNK>':
				self._unk = idx
			self._word_to_id[word] = idx
	
	def word_to_id(self, word):
		return self._word_to_id.get(word, self._unk)
	
	@property
	def unk(self):
		return self._unk
	
	def size(self):
		return len(self._word_to_id)
	
	def sentence_to_id(self, sentence):
		word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
		return word_ids

# 定义类别封装
class CategoryDict:
	def __init__(self, filename):
		self._category_to_id = {}
		with open(filename, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		for line in lines:
			category = line.strip('\n')
			idx = len(self._category_to_id)
			self._category_to_id[category] = idx
	
	def size(self):
		return len(self._category_to_id)
	
	def category_to_id(self, category):
		if category not in self._category_to_id:
			raise Exception('%s is not in our category list' % category)
		return self._category_to_id[category]


vocab = Vocab(vocab_file, hps.num_word_threshold)
vocab_size = vocab.size()
tf.logging.info('vocab size: %d' % vocab_size)

category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()
tf.logging.info('num classes: %d' % num_classes)

# 定义数据集封装
class TextDataSet:
	def __init__(self, filename, vocab, category_vocab, num_timesteps):
		self._vocab = vocab
		self._category_vocab = category_vocab
		self._num_timesteps = num_timesteps
		# matrix
		self._inputs = []
		# vector
		self._outputs = []
		self._indicator = 0
		self._parse_file(filename)
	
	def _parse_file(self, filename):
		tf.logging.info('Loading data from %s', filename)
		with open(filename, 'r', encoding = 'utf-8') as f:
			lines = f.readlines()
		for line in lines:
			label, content = line.strip('\n').split('\t')
			id_label = self._category_vocab.category_to_id(label)
			id_words = self._vocab.sentence_to_id(content)
			id_words = id_words[0:self._num_timesteps]
			padding_num = self._num_timesteps - len(id_words)
			id_words = id_words + [self._vocab.unk for i in range(padding_num)]
			self._inputs.append(id_words)
			self._outputs.append(id_label)
		self._inputs = np.array(self._inputs, dtype = np.int32)
		self._outputs = np.array(self._outputs, dtype = np.int32)
		self._random_shuffle()
		
	def _random_shuffle(self):
		p = np.random.permutation(len(self._inputs))
		self._inputs = self._inputs[p]
		self._outputs = self._outputs[p]
	
	def next_batch(self, batch_size):
		end_indicator = self._indicator + batch_size
		if end_indicator > len(self._inputs):
			self._random_shuffle()
			self._indicator = 0
			end_indicator = batch_size
		if  end_indicator > len(self._inputs):
			raise Exception('batch_size: %d if too large' % batch_size)
		
		batch_inputs = self._inputs[self._indicator: end_indicator]
		batch_outputs = self._outputs[self._indicator: end_indicator]
		self._indicator = end_indicator
		return batch_inputs, batch_outputs

train_dateset = TextDataSet(train_file, vocab, category_vocab, hps.num_timesteps)
val_dateset = TextDataSet(val_file, vocab, category_vocab, hps.num_timesteps)
test_dateset = TextDataSet(test_file, vocab, category_vocab, hps.num_timesteps)


# 构建计算图
def create_model(hps, vocab_size, num_classes):
	num_timesteps = hps.num_timesteps
	batch_size = hps.batch_size
	
	inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
	outputs = tf.placeholder(tf.int32, (batch_size, ))
	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
	
	global_step = tf.Variable(
			tf.zeros([], tf.int64), name = 'global_step', trainable = False)
	
	embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
	with tf.variable_scope('embedding', initializer = embedding_initializer):
		embeddings = tf.get_variable(
				'embeddings',
				[vocab_size, hps.num_embedding_size],
				tf.float32)
		# [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
		embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)
	
	scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
	lstm_init = tf.random_uniform_initializer(-scale, scale)
	with tf.variable_scope('lstm_nn', initializer = lstm_init):
		cells = []
		for i in range(hps.num_lstm_layers):
			cell = tf.contrib.rnn.BasicLSTMCell(
					hps.num_lstm_nodes[i],
					state_is_tuple = True)
			cell = tf.contrib.rnn.DropoutWrapper(
					cell,
					output_keep_prob = keep_prob)
			cells.append(cell)
		cell = tf.contrib.rnn.MultiRNNCell(cells)
		
		initial_state = cell.zero_state(batch_size, tf.float32)
		# rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
		rnn_outputs, _ = tf.nn.dynamic_rnn(
				cell,
				embed_inputs,
				initial_state = initial_state)
		last = rnn_outputs[:, -1, :]

	fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
	with tf.variable_scope('fc', initializer = fc_init):
		fc1 = tf.layers.dense(
				last,
				hps.num_fc_nodes,
				activation = tf.nn.relu,
				name = 'fc1')
		fc1_dropout = tf.contrib.layers.dropout(
				fc1,
				keep_prob)
		logits = tf.layers.dense(
				fc1_dropout,
				num_classes,
				name = 'fc2')
	
	with tf.name_scope('metrics'):
		softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits = logits,
				labels = outputs)
		loss = tf.reduce_mean(softmax_loss)
		# [0, 1, 5, 4, 2] -> argmax: 2
		y_pred = tf.argmax(
				tf.nn.softmax(logits),
				1,
				output_type = tf.int32)
		correct_pred = tf.equal(outputs, y_pred)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	with tf.name_scope('train_op'):
		tvars = tf.trainable_variables()
		for var in tvars:
			tf.logging.info('variable name: %s' % (var.name))
		grads, _ = tf.clip_by_global_norm(
				tf.gradients(loss, tvars),
				hps.clip_lstm_grads)
		optimizer = tf.train.AdamOptimizer(hps.learning_rate)
		train_op = optimizer.apply_gradients(
				zip(grads, tvars),
				global_step = global_step)
	
	return ((inputs, outputs, keep_prob),(loss, accuracy), (train_op, global_step))

placeholders, metrics, others = create_model(
		hps, vocab_size, num_classes)

inputs, outputs, keep_prob = placeholders
loss, accurary = metrics
train_op, global_step = others


# 定义训练流程




