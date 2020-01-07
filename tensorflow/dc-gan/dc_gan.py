# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:47:50 2020

@author: gongyue
"""

"""
1. Data provider
	a. Image data
	b. random vector
2. Build compute graph
	a. generator
	b. discriminator
	c.DCGAN
3. training process
"""

import os
import sys
import tensorflow as tf
from tensorflow import logging
from tensorflow import gfile
import pprint
import pickle as cPickle
import numpy as np
import random
import math
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data\\', one_hot=True)

output_dir = 'local_run'
if not gfile.Exists(output_dir):
	gfile.MakeDirs(output_dir)

# 定义超参数
def get_default_params():
	return tf.contrib.training.HParams(
			z_dim= 100,
			init_conv_size = 4,
			g_channels = [128, 64, 32, 1],
			d_channels = [32, 64, 128, 256],
			batch_size = 128,
			learning_rate = 0.002,
			beta1 = 0.5,
			img_size = 32)

hps = get_default_params()

class MnistData(object):
	def __init__(self, mnist_train, z_dim, img_size):
		self._data = mnist_train
		self._example_num = len(self._data)
		self._z_data = np.random.standard_normal((self._example_num, z_dim))
		self._indicator = 0
		self._resize_mnist_img(img_size)
		self._random_shuffle()
	
	def _random_shuffle(self):
		p = np.random.permutation(self._example_num)
		self._z_data = self._z_data[p]
		self._data = self._data[p]
	
	def _resize_mnist_img(self, img_size):
		"""
		Resize mnist image to goal img_size
		How?
		1. numpy -> PIL img
		2. PIL img -> resize
		3. PIL img -> numpy
		"""
		data = np.asarray(self._data * 255, np.uint8)
		data = data.reshape(self._example_num, 28, 28)
		new_data = []
		for i in range(self._example_num):
			img = data[i]
			img = Image.fromarray(img)
			img = img.resize((img_size, img_size))
			img = np.asarray(img)
			img = img.reshape((img_size, img_size, 1))
			new_data.append(img)
		new_data = np.asarray(new_data, dtype=np.float32)
		new_data = new_data / 127.5 -1
		self._data = new_data
	
	def next_batch(self, batch_size):
		end_indicator = self._indicator + batch_size
		if end_indicator > self._example_num:
			self._random_shuffle()
			self._indicator = 0
			end_indicator = self._indicator + batch_size
		assert end_indicator < self._example_num
		
		batch_data = self._data[self._indicator: end_indicator]
		batch_z = self._z_data[self._indicator: end_indicator]
		self._indicator = end_indicator
		return batch_data, batch_z

mnist_data = MnistData(mnist.train.images, hps.z_dim, hps.img_size)
batch_data, batch_z = mnist_data.next_batch(5)

def conv2d_transpose(inputs, out_channel, name, training, with_nn_relu=True):
	"""Wrapper of conv2d transpose."""
	with tf.variable_scope(name):
		conv2d_trans = tf.layers.conv2d_transpose(
				inputs,
				out_channel,
				[5, 5],
				strides = (2, 2),
				padding = 'SAME')
	
	if with_nn_relu:
		bn = tf.layers.batch_normalization(
				conv2d_trans,
				training = training)
		return tf.nn.relu(bn)
	else:
		return conv2d_trans

class Generator(object):
	def __init__(self, channels, init_conv_size):
		self._channels = chennels
		self._init_conv_size


