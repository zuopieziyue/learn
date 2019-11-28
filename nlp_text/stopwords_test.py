# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:23:54 2019

@author: gongyue
"""

import jieba

stop_word_lst = set()
with open('stopword', 'r', encoding='utf-8') as f:
	stop_word_lst = [word.strip() for word in f.readlines()]
print(stop_word_lst)

s1 = '这只皮靴号码大了。那只号码合适'
s1_direct_cut = [i for i in jieba.cut(s1, cut_all=True)]
s1_cut = [i for i in jieba.cut(s1, cut_all=True) if i not in stop_word_lst]

s2 = '这只皮靴号码不小。那只更合适'
s2_direct_cut = [i for i in jieba.cut(s2, cut_all=True)]
s2_cut = [i for i in jieba.cut(s2, cut_all=True) if i not in stop_word_lst]
print(s1_direct_cut)
print(s1_cut)
print(s2_direct_cut)
print(s2_cut)

word_set = set(s1_cut).union(set(s2_cut))
print(word_set)