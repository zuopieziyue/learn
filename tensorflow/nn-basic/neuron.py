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
	

	







