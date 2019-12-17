# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:52:21 2019

@author: gongyue
"""

import tensorflow as tf
import os
import pickle
import numpy as np

CIFAR_DIR = 'cifar-10-batches-py'
print(os.listdir(CIFAR_DIR))

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
			for item, label in zip(data, labels):
				if label in [0, 1]:
					all_data.append(item)
					all_labels.append(label)
		
		self._data = np.vstack(all_data)
		self._data = self._data / 127.5 - 1
		self._labels = np.hstack(all_labels)


x = tf.placeholder(tf.float32, [None, 3072])
# [None]
y = tf.placeholder(tf.int64, [None])









