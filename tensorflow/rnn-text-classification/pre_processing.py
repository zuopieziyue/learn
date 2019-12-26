# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:43:52 2019

@author: gongyue
"""

# 分词
# 词语 -> id
#  matrix -> ([V], embed_size)
#  词语A -> id(5)
#  词表

# lable -> id

import sys
import os
import jieba

# input_file
train_file = 'text_classification_data\\cnews.train.txt'
val_file = 'text_classification_data\\cnews.val.txt'
test_file = 'text_classification_data\\cnews.test.txt'

# output_file
seg_train_file = 'text_classification_data\\cnews.train.seg.txt'
seg_val_file = 'text_classification_data\\cnews.val.seg.txt'
seg_test_file = 'text_classification_data\\cnews.test.seg.txt'

vocab_file = 'text_classification_data\\cnews.vocab.txt'
category_file = 'text_classification_data\\cnews.category.txt'


def generate_seg_file(inputfile, output_seg_file):
	'''Segment the sentences in each line in input_file'''
	with open(inputfile, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	with open(output_seg_file, 'w', encoding='utf-8') as f:
		for line in lines:
			label, content = line.strip('\r\n').split('\t')
			word_iter = jieba.cut(content)
			word_content = ''
			for word in word_iter:
				word = word.strip(' ')
				if word != '':
					word_content += word + ' '
			out_line = '%s\t%s\n' % (label, word_content.strip(' '))
			f.write(out_line)


def generate_vocab_file(input_seg_file, output_vocab_file):
	with open(input_seg_file, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	word_dict = {}
	for line in lines:
		label, content = line.strip('\n').split('\t')
		for word in content.split():
			word_dict.setdefault(word, 0)
			word_dict[word] += 1
	
	# [(word, frequency), ..., ()]
	sorted_word_dict = sorted(word_dict.items(), key = lambda d:d[1], reverse = True)
	with open(output_vocab_file, 'w', encoding='utf-8') as f:
		f.write('<UNK>\t1000000\n')
		for item in sorted_word_dict:
			f.write('%s\t%d\n' % (item[0], item[1]))


def generate_category_dict(input_file, category_file):
	with open(input_file, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	category_dict = {}
	for line in lines:
		label, content = line.strip('\n').split('\t')
		category_dict.setdefault(label, 0)
		category_dict[label] += 1
	category_number = len(category_dict)
	with open(category_file, 'w', encoding='utf-8') as f:
		for category in category_dict:
			line = '%s\n' % category
			print('%s\t%d' % (category, category_dict[category]))
			f.write(line)

# 分词
#generate_seg_file(train_file, seg_train_file)
#generate_seg_file(val_file, seg_val_file)
#generate_seg_file(test_file, seg_test_file)

# 词表
#generate_vocab_file(seg_train_file, vocab_file)

# 类别
#generate_category_dict(train_file, category_file)




