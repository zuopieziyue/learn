# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:26:49 2019

@author: gongyue
"""

import numpy as np
import jieba
import math

a = np.array([1,0,1])
b = np.array([1,1,0])

sum = 0
for i,j in zip(a,b):
	sum += i*j
print(sum)
print(a.dot(b))
	

s1 = '这只皮靴号码大了。那只号码合适'
s1_cut = [i for i in jieba.cut(s1, cut_all=True) if i !='']
s2 = '这只皮靴号码不小。那只更合适'
s2_cut = [i for i in jieba.cut(s2, cut_all=True) if i !='']
print(s1_cut)
print(s2_cut)
word_set = set(s1_cut).union(set(s2_cut))
print(word_set)

word_dict = dict()
i = 0
for word in word_set:
	word_dict[word] = i
	i += 1
print(word_dict)

s1_cut_code = [0]*len(word_dict)
for word in s1_cut:
	s1_cut_code[word_dict[word]] += 1

s2_cut_code = [0]*len(word_dict)
for word in s2_cut:
	s2_cut_code[word_dict[word]] += 1

print(s1_cut_code)
print(s2_cut_code)

s2_np = np.array([i*i for i in s2_cut_code])
s2_np_sum = np.sum(s2_np)

s2_e = [i/math.sqrt(s2_np_sum) for i in s2_cut_code]
s2_e_sum = np.sum([i*i for i in s2_e])

print(s2_e)
print(s2_e_sum)






