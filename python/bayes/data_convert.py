# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:59:51 2019

@author: Administrator
"""

import os
import random
import sys

file_path = 'raw_data\\'

TrainingPercent = 0.8
train_out_file = open('raw_data\\data.train', 'w')
test_out_file = open('raw_data\\data.test', 'w')
label_dict = {'business':0, 'yule':1, 'it':2, 'sports':3, 'auto':4}
i = 0
tag = 0

for filename in os.listdir(file_path):
	if filename.find('business') != -1:
		tag = label_dict['business']
	elif filename.find('yule') != -1:
		tag = label.dict['yule']
	elif filename.find('it') != -1:
		tag = label.dict['it']
	elif filename.find('sports') != -1:
		tag = label.dict['sports']
	elif filename.find('auto') != -1:
		tag = label.dict['auto']

	i += 1
	rd = random.random()
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
		
	
	
		
		
		
		
		
		



	
		
		
		
		
		
		
		