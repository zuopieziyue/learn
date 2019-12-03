# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:59:51 2019

@author: Administrator
"""

import os
import random
import sys

file_path = 'raw_data\\'
TrainOutFilePath = 'mid_data\\data.train'
TestOutFilePath = 'mid_data\\data.test'

TrainingPercent = 0.8
train_out_file = open(TrainOutFilePath, 'w', encoding="utf-8")
test_out_file = open(TestOutFilePath, 'w', encoding="utf-8")
label_dict = {'business':0, 'yule':1, 'it':2, 'sports':3, 'auto':4}

WordIDDic = dict()
WordList = []

def convert_data():
	i = 0
	tag = 0
	
	for filename in os.listdir(file_path):
		if filename.find('business') != -1:
			tag = label_dict['business']
		elif filename.find('yule') != -1:
			tag = label_dict['yule']
		elif filename.find('it') != -1:
			tag = label_dict['it']
		elif filename.find('sports') != -1:
			tag = label_dict['sports']
		elif filename.find('auto') != -1:
			tag = label_dict['auto']
	
		i += 1
		rd = random.random()
		outfile = test_out_file
	
		if rd < TrainingPercent:
			outfile = train_out_file
		if i%100 == 0:
			print(i, "files processed!")
		
		infile = open(os.path.join(file_path, filename), 'r', encoding='utf-8')
		outfile.write(str(tag) + ' ')
		
		content = infile.read().strip()
		words = content.replace('\n', ' ').split(' ')
		for word in words:
			if len(word.strip()) < 1:
				continue
			if word not in WordIDDic:
				WordList.append(word)
				WordIDDic[word] = len(WordList)
			outfile.write(str(WordIDDic[word]) + ' ')
		outfile.write('#' + filename + '\n')
		infile.close()
		
		print(i, "files loaded!")
		print(len(WordList), "uniqe words found!")


if __name__ == '__main__':
	convert_data()
	train_out_file.close()
	test_out_file.close()
