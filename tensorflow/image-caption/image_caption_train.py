# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 23:33:57 2020

@author: Administrator
"""

"""
1. Data Generator
	a. Load vocab
	b. Loads image features
	c. provide data for training.
2. Builds image caption model.
3. Trains the model.
"""

import os
import sys
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
import pprint
import pickle as cPickle
import numpy as np
import math

tf.logging.set_verbosity( tf.compat.v1.logging.INFO)

input_description_file = 'image_caption_data\\results_20130124.token'
input_img_feature_dir = 'image_caption_data\\feature_extraction_inception_v3'
input_vocab_file = 'image_caption_data\\vocab.txt'
output_dir = 'image_caption_data\\local_run'

if not gfile.Exists(output_dir):
	gfile.MakeDirs(output_dir)

def get_default_params():
	return tf.contrib.training.HParams(
			num_vocab_word_threshold = 3,
			num_embedding_nodes = 32,
			num_timesteps = 10,
			num_lstm_nodes = [64, 64],
			num_lstm_layers = 2,
			num_fc_nodes = 32,
			batch_size = 50,
			cell_type = 'lstm',
			clip_lstm_grads = 1.0,
			learning_rate = 0.001,
			keep_prob = 0.8,
			log_frequent = 100,
			save_frequent = 1000,
	)

hps = get_default_params()

class Vocab(object):
	def __init__(self, filename, word_num_threshold):
		self._id_to_word = {}
		self._word_to_id = {}
		self._unk = -1
		self._eos = -1
		self._word_num_threshold = word_num_threshold
		self._read_dict(filename)
	
	def _read_dict(self, filename):
		with gfile.GFile(filename, 'r') as f:
			lines = f.readlines()
		for line in lines:
			word, occurence = line.strip('\r\n').split('\t')
			occurence = int(occurence)
			if word != '<UNK>' and occurence < self._word_num_threshold:
				continue
			idx = len(self._id_to_word)
			if word == '<UNK>':
				self._unk = idx
			elif word == '.':
				self._eos = idx
			if idx in self._id_to_word or word in self._word_to_id:
				raise Exception('duplicate words in vocab file')
			self._word_to_id[word] = idx
			self._id_to_word[idx] = word
	
	@property
	def unk(self):
		return self._unk

	@property
	def eos(self):
		return self._eos
	
	def word_to_id(self, word):
		return self._word_to_id.get(word, self.unk)
	
	def id_to_word(self, cur_id):
		return self._id_to_word.get(cur_id, '<UNK>')
	
	def size(self):
		return len(self._word_to_id)
	
	def encode(self, sentence):
		word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split(' ')]
		return word_ids
	
	def decode(self, sentence_id):
		words = [self.id_to_word(word_id) for word_id in sentence_id]
		return ' '.join(words)
	
def parse_token_file(token_file):
	"""Parses token file."""
	img_name_to_tokens = {}
	with gfile.GFile(token_file, 'r') as f:
		lines = f.readlines()
	for line in lines:
		img_id, description = line.strip('\r\n').split('\t')
		img_name, _ = img_id.split('#')
		img_name_to_tokens.setdefault(img_name, [])
		img_name_to_tokens[img_name].append(description)
	return img_name_to_tokens

def convert_token_to_id(img_name_to_tokens, vocab):
	"""Convert tokens of each description of imgs to id."""
	img_name_to_token_ids = {}
	for img_name in img_name_to_tokens:
		img_name_to_token_ids.setdefault(img_name, [])
		descriptions = img_name_to_tokens[img_name]
		for description in descriptions:
			token_ids = vocab.encode(description)
			img_name_to_token_ids[img_name].append(token_ids)
	return img_name_to_token_ids

vocab = Vocab(input_vocab_file, hps.num_vocab_word_threshold)
vocab_size = vocab.size()
logging.info('vocab_size: %d' % vocab_size)

img_name_to_tokens = parse_token_file(input_description_file)
img_name_to_token_ids = convert_token_to_id(img_name_to_tokens, vocab)

logging.info('num of all images: %d' % len(img_name_to_tokens))
pprint.pprint(list(img_name_to_tokens.keys())[0:10])
pprint.pprint(img_name_to_tokens['2778832101.jpg'])
logging.info('num of all images: %d' % len(img_name_to_token_ids))
pprint.pprint(list(img_name_to_token_ids.keys())[0:10])
pprint.pprint(img_name_to_token_ids['2778832101.jpg'])


class ImageCaptionData(object):
	def __init__(
			self,
			img_name_to_token_ids,
			img_feature_dir,
			num_timesteps,
			vocab,
			deterministic = False):
		self._vocab = vocab
		self._all_img_feature_filepaths = []
		for filename in gfile.ListDirectory(img_feature_dir):
			self._all_img_feature_filepaths.append(os.path.join(img_feature_dir, filename))
		pprint.pprint(self._all_img_feature_filepaths)
		
		self._img_name_to_token_ids = img_name_to_token_ids
		self._num_timesteps = num_timesteps
		self._indicator = 0
		self._deterministic = deterministic
		self._img_feature_filenames = []
		self._img_feature_data = []
		self._load_img_feature_pickle()
		if not self._deterministic:
			self._random_shuffle()
	
	def _load_img_feature_pickle(self):
		for filepath in self._all_img_feature_filepaths:
			logging.info('loading %s' % filepath)
			with gfile.GFile(filepath, 'rb') as f:
				filenames, features = cPickle.load(f, encoding='bytes')
				self._img_feature_filenames += filenames
				self._img_feature_data.append(features)
		self._img_feature_data = np.vstack(self._img_feature_data)
		origin_shape = self._img_feature_data.shape
		self._img_feature_data = np.reshape(
				self._img_feature_data,
				(origin_shape[0], origin_shape[3]))
		self._img_feature_filenames = np.asarray(self._img_feature_filenames)
		print(self._img_feature_data.shape)
		print(self._img_feature_filenames.shape)
		if not self._deterministic:
			self._random_shuffle()
	
	def size(self):
		return len(self._img_feature_filenames)
	
	def img_feature_size(self):
		return self._img_feature_data.shape[1]

	def _random_shuffle(self):
		p = np.random.permutation(self.size())
		self._img_feature_filenames = self._img_feature_filenames[p]
		self._img_feature_data = self._img_feature_data[p]
	
	def _img_desc(self, filenames):
		batch_sentence_ids = []
		batch_weights = []
		for filename in filenames:
			filename = filename.decode()
			token_ids_set = self._img_name_to_token_ids[filename]
			#chosen_token_ids = random.choice(token_ids_set)
			chosen_token_ids = token_ids_set[0]
			chosen_token_length = len(chosen_token_ids)
			
			weight = [1 for i in range(chosen_token_length)]
			if chosen_token_length >= self._num_timesteps:
				chosen_token_ids = chosen_token_ids[0:self._num_timesteps]
				weight = weight[0:self._num_timesteps]
			else:
				remaining_length = self._num_timesteps - chosen_token_length
				chosen_token_ids += [self._vocab.eos for i in range(remaining_length)]
				weight += [0 for i in range(remaining_length)]
			batch_sentence_ids.append(chosen_token_ids)
			batch_weights.append(weight)
		batch_sentence_ids = np.asarray(batch_sentence_ids)
		batch_weights = np.asarray(batch_weights)
		return batch_sentence_ids, batch_weights
	
	def next_batch(self, batch_size):
		end_indicator = self._indicator + batch_size
		if end_indicator > self.size():
			if not self._deterministic:
				self._random_shuffle()
			self._indicator = 0
			end_indicator = self._indicator + batch_size
		assert end_indicator <= self.size()
		
		batch_img_features = self._img_feature_data[self._indicator: end_indicator]
		batch_img_names = self._img_feature_filenames[self._indicator: end_indicator]
		batch_sentence_ids, batch_weights = self._img_desc(batch_img_names)

		self._indicator = end_indicator
		return batch_img_features, batch_sentence_ids, batch_weights, batch_img_names

caption_data = ImageCaptionData(img_name_to_token_ids, input_img_feature_dir, hps.num_timesteps, vocab)
img_feature_dim = caption_data.img_feature_size()
caption_data_size = caption_data.size()
logging.info('img_feature_dim: %d' % img_feature_dim)
logging.info('caption_data_size: %d' % caption_data_size)

batch_img_features, batch_sentence_ids, batch_weights, batch_img_names = caption_data.next_batch(5)
pprint.pprint(batch_img_features)
pprint.pprint(batch_sentence_ids)
pprint.pprint(batch_weights)
pprint.pprint(batch_img_names)

