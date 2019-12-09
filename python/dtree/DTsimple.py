# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:18:50 2019

@author: gongyue
"""

import numpy as np

def createDataSet():
	dataSet = [[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

def calcEnt(dataSet):
	n = len(dataSet)
	label_cnt = {}
	for featVec in dataSet:
		cur_label = featVec[-1]
		if cur_label not in label_cnt.keys():
			label_cnt[cur_label] = 0
		label_cnt[cur_label] += 1
	
	E = 0.0
	for key in label_cnt:
		
		
			
			
	
	