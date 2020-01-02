# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:06:55 2020

@author: gongyue
"""

import os
import sys
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
import pprint
import pickle as pk
import numpy as np

model_file = 'image_caption_data\\checkpoint_inception_v3\\inception_v3_graph_def.pb'
input_description_file = 'image_caption_data\\results_20130124.token'
input_img_dir = 'image_caption_data\\flick30k_images'
output_folder = 'image_caption_data\\download_inception_v3_features_bakup'

batch_size = 1000
if not gfile.Exists(output_folder):
	gfile.MakeDirs(output_folder)

def parse_token_file(token_file):
	'''
	Parses image description file.
	'''
	img_name_to_tokens = {}
	with gfile.GFile(token_file, 'r') as f:
		lines = f.readlines()
	
	for line in lines:
		img_id, description = line.strip('\r\n').split('\t')
		img_name, _ = img_id.split('#')
		img_name_to_tokens.setdefault(img_name, [])
		img_name_to_tokens[img_name].append(description)
	return img_name_to_tokens

img_name_to_tokens = parse_token_file(input_description_file)
all_img_names = img_name_to_tokens.keys()

logging.info('num of all images: %d' % len(all_img_names))
pprint.pprint(list(img_name_to_tokens.keys())[0:10])
pprint.pprint(img_name_to_tokens['2778832101.jpg'])

def load_pretrained_inception(model_file):
	with gfile.FastGFile(model_file, 'r') as f:
		graph_def = tf.
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

load_pretrained_inception(model_file)


















