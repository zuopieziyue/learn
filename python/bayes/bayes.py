# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:26:36 2019

@author: gongyue
"""
import math

'''
训练，测试数据路径
'''
TrainOutFilePath = 'mid_data\\data.train'
TestOutFilePath = 'mid_data\\data.test'

'''
模型存储路径
'''
Modelfile = 'mid_data\\bayes.Model'

DefaultFreq = 0.1
ClassDefaultProb = dict()
ClassFeatDic = dict()
ClassFeatProb = dict()
ClassFreq = dict()
ClassProb = dict()

WordDic = dict()

def Dedup(items):
	temp_dic = {}
	for item in items:
		if item not in temp_dic:
			temp_dic[item] = True
	return temp_dic.keys()

def LoadData():
	i = 0
	infile = open(TrainOutFilePath, 'r', encoding='utf-8')
	sline = infile.readline().strip()
	while len(sline) > 0:
		pos = sline.find("#")
		if pos > 0:
			sline = sline[:pos].strip()
		words = sline.split(" ")
		if len(words) < 1:
			print("Format error!")
		classid = int(words[0])
		if classid not in ClassFeatDic:
			ClassFeatDic[classid] = dict()
			ClassFeatProb[classid] = dict()
			ClassFreq[classid] = 0
		ClassFreq[classid] += 1
		words = words[1:]
		
		#贝努力分布，需要去除重复词
		#words = Dedup(words)
		
		#如果是多项式分布，不用做处理
		for word in words:
			if len(word) < 1:
				continue
			wid = int(word)
			if wid not in WordDic:
				WordDic[wid] = 1
			if wid not in ClassFeatDic[classid]:
				ClassFeatDic[classid][wid] = 1
			else:
				ClassFeatDic[classid][wid] += 1
		
		i += 1
		sline = infile.readline().strip()
	infile.close()
	print(i, "instance loaded!")
	print(len(ClassFreq), "classes!", len(WordDic), "words!")

'''
在原有统计count做成概率，增加平滑处理
'''
def ComputeModel():
	mysum = 0.0
	for freq in ClassFreq.values():
		mysum += freq
	for classid in ClassFreq.keys():
		ClassProb[classid] = float(ClassFreq[classid])/mysum
	for classid in ClassFeatDic.keys():
		mysum = 0.0
		for wid in ClassFeatDic[classid].keys():
			mysum += ClassFeatDic[classid][wid]
		
		#平滑处理 （+1平滑，拉普拉斯平滑）
		newsum = float(mysum + len(WordDic)) * DefaultFreq
		for wid in ClassFeatDic[classid].keys():
			ClassFeatProb[classid][wid] = float(ClassFeatDic[classid][wid] + DefaultFreq) / newsum
		ClassDefaultProb[classid] = float(DefaultFreq)/newsum

def SaveModel():
	outfile = open(Modelfile, 'w', encoding='utf-8')
	for classid in ClassFreq.keys():
		outfile.write(str(classid) + ' ' + str(ClassProb[classid]) + ' ' + ClassDefaultProb + ' ')
	outfile.write('\n')
	
	for classid in ClassFeatDic.keys():
		for wid in ClassFeatProb[classid].keys():
			outfile.write(str(wid) + ' ' + str(ClassFeatProb[classid][wid]) + ' ')
		outfile.write('\n')
	outfile.close()

def LoadModel():
	global WordDic
	WordDic = {}
	global ClassFeatProb
	ClassFeatProb = {}
	global ClassDefaultProb
	ClassDefaultProb = {}
	global ClassProb
	ClassProb = {}
	
	infile = open(Modelfile, 'r', encoding='utf-8')
	sline = infile.readline().strip()
	items = sline.split(' ')
	if len(items) < 3*5:
		print("Model format error!")
		return
	i = 0
	while i < len(items):
		classid = int(items[i])
		ClassFeatProb[classid] = {}
		i += 1
		if i >= len(items):
			print("Model format error!")
			return
		ClassProb[classid] = float(items[i])
		i += 1
		ClassDefaultProb[classid] = float(items[i])
		i += 1
	
	for classid in ClassProb.keys():
		sline = infile.readline().strip()
		items = sline.split(' ')
		i = 0
		for item in items:
			wid = int(items[i])
			if wid not in WordDic:
				WordDic[wid] = 1
			i += 1
			ClassFeatProb[classid][wid] = float(items[i])
			i += 1
	infile.close()
	print(len(ClassProb), "classes!", len(WordDic), "words!")

def Predict():
	global WordDic
	global ClassFeatProb
	global ClassDefaultProb
	global ClassProb
	
	true_label_list = []
	pred_label_list = []
	
	infile = open(TestOutFilePath, 'r', encoding='utf-8')
	sline = infile.readline().strip()
	scoreDic = {}
	iline = 0
	while len(sline) > 0:
		iline += 1
		if iline%10 == 0:
			print(iline, "lines finished!")
		pos = sline.find("#")
		if pos > 0:
			sline = sline[:pos].strip()
		words = sline.split(" ")
		classid = int(words[0])
		true_label_list.append(classid)
		words = words[1:]
		
		# 取先验概率
		for classid in ClassProb.keys():
			scoreDic[classid] = math.log(ClassProb[classid])
		
		# 不在字典中的词丢掉
		for word in words:
			if len(word) < 1:
				continue
			wid = int(word)
			if wid not in WordDic:
				continue
			
			for classid in ClassProb.keys():
				# 在字典中没有在该类别中出现的词，用该类别默认概率计算
				if wid not in ClassFeatProb[classid]:
					scoreDic[classid] += math.log(ClassDefaultProb[classid])
				else:
					scoreDic[classid] += math.log(ClassFeatProb[classid][wid])
		
		maxProb = max(scoreDic.values())
		for classid in scoreDic.keys():
			if scoreDic[classid] == maxProb:
				pred_label_list.append(classid)
		sline = infile.readline().strip()
	
	infile.close()
	print(len(true_label_list), len(pred_label_list))
	return true_label_list, pred_label_list

def Evaluate(true_list, pred_list):
	accuracy = 0
	i = 0
	while i < len(true_list):
		if pred_list[i] == true_list[i]:
			accuracy += 1
		i += 1
	accuracy = float(accuracy)/float(len(true_list))
	print('Accuracy: ', accuracy)


if __name__=="__main__":
	LoadData()
	ComputeModel()
	true_list, pred_list = Predict()
	Evaluate(true_list, pred_list)