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
import pickle as pk
import numpy as np
import math

input_description_file = 'image_caption_data\\results_20130124.token'
input_img_feature_dir = 'image_caption_data\\feature_extraction_inception_v3'
input_vocab_file = 'image_caption_data\\vocab.txt'
output_dir = 'image_caption_data\\local_run'

if not gfile.Exists(output_dir):
	gfile.MakeDirs(output_dir)

def get_default_params():
	




