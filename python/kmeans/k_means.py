# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:44:15 2019

@author: gongyue
"""

import os
import math
import random
import operator

#参数
K = 10
threshold = 1e-6
WCSS = 0.0
new_WCSS = 1
ITER_MAX = 20

#文件路径
file_path = 'data\\'

#标签字典
label_dict = {'bussiness':0, 'yule':1, 'it':2, 'sports':3, 'auto':4}

i = 0
word_dict = dict() # 对word进行编码
def load_data():
	doc_list = []
	doc_dict = dict()
	i = 0
	for filename in os.listdir(file_path):
		doc_name = filename.split('.')[0]
		doc_list.append(doc_name)
		if i%100==0:
			print(i,'files loaded!!')
		with open(file_path+'/'+filename,'r',encoding='utf-8') as f:
			word_freq = dict()  # tf
			for line in f.readlines():
				words = line.strip().split(' ')
				for word in words:
					if len(word.strip()) < 1:
						continue
					if word_dict.get(word,-1)==-1:
						word_dict[word] = len(word_dict)
					wid = word_dict.get(word,-1)
					if word_freq.get(wid,-1)==-1:
						word_freq[wid] = 1
					else:
						word_freq[wid] += 1

			doc_dict[doc_name] = word_freq
		i += 1
	return doc_dict,doc_list

def idf(doc_dict):
	word_idf = {}
	#统计doc freq
	for doc in doc_dict.keys():
		for word in doc_dict[doc].keys():
			if word_idf.get(word, -1) == -1:
				word_idf[word] = 1
			else:
				word_idf[word] += 1
	doc_num = len(doc_dict)
	#计算idf
	for word in word_idf.keys():
		word_idf[word] = math.log(doc_num/(word_idf[word]+1))
	return word_idf

def doc_tf_idf():
	doc_dict, doc_list = load_data()
	word_idf = idf(doc_dict=doc_dict)
	
	for doc in doc_list:
		for word in doc_dict[doc].keys():
			doc_dict[doc][word] = doc_dict[doc][word] * word_idf[word]
	return doc_dict, doc_list

def init_K(doc_dict, doc_list):
	center_dict = dict()
	k_doc_list = random.sample(doc_list, K)
	i = 0
	for doc_name in k_doc_list:
		center_dict[i] = doc_dict[doc_name]
		i += 1
	return center_dict

def compute_dis(doc1, doct1_dict, doc2, doc2_dict):
	mysum = 0.0
	words = set([i for i in doct1_dict[doc1].keys()]).union(set([i for i in doc2_dict[doc2].keys()]))
	for wid in words:
		d = doct1_dict[doc1].get(wid,0.0) - doc2_dict[doc2].get(wid,0.0)
		mysum += d*d
	return mysum

def compute_center(doc_list,doc_dict):
	tmp_center = dict()
	i = 0
	for doc in doc_list:
		for wid in doc_dict[doc].keys():
			if tmp_center.get(wid,-1) == -1:
				tmp_center[wid] = doc_dict[doc][wid]
			else:
				tmp_center[wid] += doc_dict[doc][wid]
		i += 1
	for wid in tmp_center.keys():
		tmp_center[wid] /= i
	return tmp_center

def all_k_dist(doc_list, doc_dict, k, k_dict):
	mysum = 0.0
	for doc in doc_list:
		tmp_k_dict = {k:k_dict}
		mysum += compute_dis(doc, doc_dict, k, tmp_k_dict)
	return mysum

if __name__=='__main__':
	doc_dict, doc_list = doc_tf_idf()
	center_dict = init_K(doc_dict, doc_list)
	doc_k = dict(zip(doc_list,[0 for i in range(len(doc_list))]))
	
	iter_num = 0
	Center_mv = 1
	print('start train!!')
	while new_WCSS - WCSS > threshold and iter_num < ITER_MAX and Center_mv > threshold:
		k_doc = dict()
		for doc in doc_list:
			tmp_select_k = dict()
			for k in center_dict.keys():
				tmp_select_k[k] = compute_dis(doc, doc_dict, k, center_dict)
			(k, val) = sorted(tmp_select_k.items(), key=operator.itemgetter(1))[0]
			doc_k[doc] = k
			if k_doc.get(k,-1) == -1:
				k_doc[k] = [doc]
			else:
				k_doc[k].append(doc)
			
		#step 2
		Center_mv = 0
		WCSS = new_WCSS
		new_WCSS = 0
		for k in k_doc.keys():
			tmp_k_center = compute_center(k_doc[k], doc_dict)
			tmp_new_k_center = {k:tmp_k_center}
			Center_mv += compute_dis(k,center_dict,k,tmp_new_k_center)
			new_WCSS += all_k_dist(doc_list, doc_dict, k, center_dict[k])
			center_dict[k] = tmp_k_center
		print(iter_num)
		iter_num += 1


