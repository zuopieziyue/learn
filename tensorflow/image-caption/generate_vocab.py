# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:05:32 2020

@author: gongyue
"""

import os
import sys
import pprint

input_description_file = 'image_caption_data\\results_20130124.token'
output_vocab_file = 'image_caption_data\\vocab.txt'

def count_vocab(input_description_file):
	'''
	Generate vocabulary.
	In addition, count distribution of length of sentence
	and max length if image distribution.
	'''
	with open(input_description_file, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	max_length_of_sentences = 0
	length_dict = {}
	vocab_dict = {}
	for line in lines:
		image_id, description = line.strip('\n').split('\t')
		words = description.strip(' ').split()
		max_length_of_sentences = max(max_length_of_sentences, len(words))
		length_dict.setdefault(len(words), 0)
		length_dict[len(words)] += 1
		
		for word in words:
			vocab_dict.setdefault(word, 0)
			vocab_dict[word] += 1
	print (max_length_of_sentences)
	pprint.pprint(length_dict)
	return vocab_dict
		
vocab_dict = count_vocab(input_description_file)

sorted_vocab_dict = sorted(vocab_dict.items(), key = lambda d: d[1], reverse = True)
with open(output_vocab_file, 'w', encoding='utf-8') as f:
	f.write('<UNK>\t100000000\n')
	for item in sorted_vocab_dict:
		f.write('%s\t%d\n' % item)
