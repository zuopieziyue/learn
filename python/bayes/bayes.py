# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:26:36 2019

@author: gongyue
"""

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





if __name__=="__main__":
	LoadData()
	ComputeModel()