# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:17:56 2019

@author: gongyue
"""
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF,StringIndexer,Tokenizer,IDF
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
df_seg_arr = tokenizer.transform(df_seg).select('words','label')
#print(df_seg_arr.head(5))

#切词之后的文本特征的处理
tf = HashingTF(numFeatures=1<<18, binary=False, inputCol='words', outputCol='rawfeatures')
df_tf = tf.transform(df_seg_arr).select('rawfeatures','label')
#print(df_tf.head(5))

idf = IDF(inputCol='rawfeatures', outputCol='features')
idfModel = idf.fit(df_tf)
df_tfidf = idfModel.transform(df_tf)
#print(df_tfidf.head(5))

#label数据的处理
stringIndexer = StringIndexer(inputCol='label', outputCol='indexed', handleInvalid='error')
indexer = stringIndexer.fit(df_tfidf)
df_tfidf_lab = indexer.transform(df_tfidf).select('features','indexed')
#print(df_tfidf_lab.head(5))

#切分训练集和测试集
splits = df_tfidf_lab.randomSplit([0.7,0.3], 1234)
train = splits[0]
test = splits[1]

#定义模型
nb = NaiveBayes(featuresCol='features',
				labelCol='indexed',
				predictionCol='prediction',
				probabilityCol='probability',
				rawPredictionCol='rawPrediction',
				smoothing=1.0,
				modelType='multinomial')

#模型训练
model = nb.fit(train)

#测试集预测
predictions = model.transform(test)
#print(predictions.head(5))

#计算准确率
evaluator = MulticlassClassificationEvaluator(
				labelCol='indexed',
				predictionCol='prediction',
				metricName='accuracy')

accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))









