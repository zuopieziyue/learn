# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:18:50 2019

@author: gongyue
"""
import numpy as np
import math

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
		prob = float(label_cnt[key])/n
		E -= prob*math.log(prob)
	return E

def splitDataSet(dataset, i, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[i] == value:
			reducedFeatVec = featVec[:i]
			reducedFeatVec.extend(featVec[i+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	# i相当于年龄
	for i in range(numFeatures):
		# 获取当前年龄这一列的所有值
		featList = [example[i] for example in dataSet]
		# 获取青年，中年，老年。三个不同值的唯一值
		uniqueVals = set(featList)
		newEntropy = 0.0
		# value相当于青年，中年，老年
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet,i,value)
			# 青年中年老年在总（上一次划分的数据集）样本中的占比
			prob = len(subDataSet)/float(len(dataSet))
			# 划分之后的数据的信息熵Entropy
			newEntropy += prob*calcEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
		return bestFeature

def majorityCnt(classList):
	d = dict()
	for c in classList:
		if d.get(c, -1) == -1:
			d[c] = 0
		else:
			d[c] += 1
	return max(d.items(), key=lambda x: x[1])[0]

def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	# 判断类别列表中是否只有一个值，如果是表示已经是“纯”了
	if classList.count(classList[0] == len(classList)):
		return classList[0]
	# 如果数据维度为1，要基于最后一个特征进行划分,相当于接下来没有特征了，特征列表为空
	if len(dataSet[0]) == 1:
		# 返回样本中类别数据最多的一个类别
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	
	# 初始化时，用字典作为树
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	# 用特征值划分数据集
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
	return myTree

dataSet, labels = createDataSet()
treeModel = createTree(dataSet, labels)
print(treeModel)



