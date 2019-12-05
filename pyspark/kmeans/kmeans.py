# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:07:36 2019

@author: gongyue
"""

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF,Tokenizer,IDF
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans

datafile_path = "D:\\github\\learn\\pyspark\\bayes\\allfiles.txt"

#创建SparkSession
se = SparkSession.builder.config(conf = SparkConf()).getOrCreate()
sc = se.sparkContext

#读取文件
allfile_rdd = sc.textFile(datafile_path)

#将RDD格式的数据转换为DF
allfile_list = allfile_rdd.collect()
allfile_list_split = [line.split('##@@##') for line in allfile_list]
df_seg = se.createDataFrame(allfile_list_split)
df_seg = df_seg.withColumnRenamed('_1','seg').withColumnRenamed('_2','label')
#print(df_seg.head(5))

#将分词做成ArrayType()
tokenizer = Tokenizer(inputCol='seg',outputCol='words')
df_seg_arr = tokenizer.transform(df_seg).select('words')
#print(df_seg_arr.head(5))

#切词之后的文本特征的处理
tf = HashingTF(numFeatures=1<<18, binary=False, inputCol='words', outputCol='rawfeatures')
df_tf = tf.transform(df_seg_arr).select('rawfeatures')
#print(df_tf.head(5))

idf = IDF(inputCol='rawfeatures', outputCol='features')
idfModel = idf.fit(df_tf)
df_tfidf = idfModel.transform(df_tf)
#print(df_tfidf.head(5))

#切分训练集和测试集
splits = df_tfidf.randomSplit([0.7,0.3], 1234)
train = splits[0]
test = splits[1]

#定义模型
kmeans = KMeans(
		featuresCol='features',
		predictionCol='prediction',
		k=6,
		initMode='k-means||',
		initSteps=5,
		tol=1e-4,
		maxIter=20,
		seed=None
		)

#模型训练
model = kmeans.fit(train)

#获取中心点
centers = model.clusterCenters()
print('clusterCenters: ')
for center in centers:
	print(center)
wssse = model.computeCost(train)
print('Within Set Sum of Squared Errors: ' + str(wssse))

