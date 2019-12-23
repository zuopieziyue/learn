# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:19:09 2019

@author: gongyue
"""

import tensorflow as tf
import os
import pickle
import numpy as np

CIFAR_DIR = 'cifar-10-batches-py'
print(os.listdir(CIFAR_DIR))


# 数据加载和数据 ===================================

def load_data(filename):
	# read data from data file
	with open(filename, 'rb') as f:
		data = pickle.load(f, encoding='bytes')
		return data[b'data'], data[b'labels']

# tensorflow.Dataset
class CifarData:
	def __init__(self, filenames, need_shuffle):
		all_data = []
		all_labels = []
		for filename in filenames:
			data, labels = load_data(filename)
			all_data.append(data)
			all_labels.append(labels)
		self._data = np.vstack(all_data)
		self._data = self._data / 127.5 - 1
		self._labels = np.hstack(all_labels)
		print(self._data.shape)
		print(self._labels.shape)

		self._num_examples = self._data.shape[0]
		self._need_shuffle = need_shuffle
		self._indicator = 0
		if self._need_shuffle:
			self._shuffle_data()
	
	def _shuffle_data(self):
		# [0,1,2,3,4,5] -> [5,3,2,4,0,1]
		p = np.random.permutation(self._num_examples)
		self._data = self._data[p]
		self._labels = self._labels[p]
	
	def next_batch(self, batch_size):
		# return batch_size examples as a batch.
		end_indicator = self._indicator + batch_size
		if end_indicator > self._num_examples:
			if self._need_shuffle:
				self._shuffle_data()
				self._indicator = 0
				end_indicator = batch_size
			else:
				raise Exception('have no more examples')
		if end_indicator > self._num_examples:
			raise Exception("batch size is larger than all examples")
		batch_data = self._data[self._indicator:end_indicator]
		batch_labels = self._labels[self._indicator:end_indicator]
		self._indicator = end_indicator
		return batch_data, batch_labels

train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)


# 计算图定义 ==============================
def inception_block(x,
					output_channel_for_each_path,
					name):
	'''
	- x: 
	- output_channel_for_each_path: eg [10, 20, 5]
	- name:
	'''
	with tf.variable_scope(name):
		conv1_1 = tf.layers.conv2d(
				x,
				output_channel_for_each_path[0],
				(1, 1),
				strides = (1, 1),
				padding = 'same',
				activation = tf.nn.relu,
				name = 'conv1_1')
		conv3_3 = tf.layers.conv2d(
				x,
				output_channel_for_each_path[1],
				(3, 3),
				strides = (1, 1),
				padding = 'same',
				activation = tf.nn.relu,
				name = 'conv3_3')
		conv5_5 = tf.layers.conv2d(
				x,
				output_channel_for_each_path[2],
				(5, 5),
				strides = (1, 1),
				padding = 'same',
				activation = tf.nn.relu,
				name = 'conv5_5')
		max_pooling = tf.layers.max_pooling2d(
				x,
				(2, 2),
				(2, 2),
				name = 'max_pooling')
	max_pooling_shape = max_pooling.get_shape().as_list()[1:]
	input_shape = x.get_shape().as_list()[1:]
	width_padding = (input_shape[0] - max_pooling_shape[0]) // 2
	height_padding = (input_shape[1] - max_pooling_shape[1]) // 2
	padded_pooling = tf.pad(
			max_pooling,
			[[0, 0], [width_padding, width_padding],[height_padding, height_padding], [0, 0]])
	concat_layer = tf.concat(
			[conv1_1, conv3_3, conv5_5, padded_pooling],
			axis = 3)
	return concat_layer


# [None， 3072]
x = tf.placeholder(tf.float32, [None, 3072])
# [None] eg: [0,5,6,3]
y = tf.placeholder(tf.int64, [None])
x_image = tf.reshape(x, [-1, 3, 32, 32])
# 32*32
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

# conv1 神经元图，feature_map, 输出图像
conv1 = tf.layers.conv2d(
		x_image,
		32, # output channel number
		(3, 3), # kernel size
		padding = 'same',
		activation = tf.nn.relu,
		name = 'conv1')
pooling1 = tf.layers.max_pooling2d(
		conv1,
		(2, 2),
		(2, 2),
		name = 'pool1')

inception_2a = inception_block(
		pooling1,
		[16, 16, 16],
		name = 'inception_2a')
inception_2b = inception_block(
		inception_2a,
		[16, 16, 16],
		name = 'inception_2b')

pooling2 = tf.layers.max_pooling2d(
		inception_2b,
		(2, 2),
		(2, 2),
		name = 'pool2')

inception_3a = inception_block(
		pooling2,
		[16, 16, 16],
		name = 'inception_3a')
inception_3b = inception_block(
		inception_3a,
		[16, 16, 16],
		name = 'inception_3b')

pooling3 = tf.layers.max_pooling2d(
		inception_3b,
		(2, 2),
		(2, 2),
		name = 'pool3')

flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
# y_ -> sofmax
# y -> one_hot
# loss = ylogy_


# indices
predict = tf.argmax(y_, 1)
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)



# 模型训练 ===========================================

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

with tf.Session() as sess:
	sess.run(init)
	for i in range(train_steps):
		batch_data, batch_labels = train_data.next_batch(batch_size)
		loss_val, acc_val, _ = sess.run(
				[loss, accuracy, train_op],
				feed_dict = {
						x: batch_data,
						y: batch_labels
						})
		
		if (i+1) % 500 == 0:
			print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i+1, loss_val, acc_val))
		if (i+1) % 5000 == 0:
			test_data = CifarData(test_filenames, False)
			all_test_acc_val = []
			for j in range(test_steps):
				test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
				test_acc_val = sess.run(
						[accuracy],
						feed_dict = {
								x: test_batch_data,
								y: test_batch_labels})
				all_test_acc_val.append(test_acc_val)
			test_acc = np.mean(all_test_acc_val)
			print('[Test] Step: %d, acc: %4.5f' % (i+1, test_acc))