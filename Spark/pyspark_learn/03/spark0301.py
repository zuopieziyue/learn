# encoding = 'utf-8'

import os
import sys
os.environ['SPARK_HOME'] = '/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0'
sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")

from pyspark import SparkConf, SparkContext

# 创建SparkConf：设置的是Spark相关的参数信息
#conf = SparkConf().setMaster('local').setAppName('spark0301')
conf = SparkConf()
# 创建SparkContext
sc = SparkContext(conf=conf)


# 业务逻辑的开发
data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)
print(distData.collect())

# stop
sc.stop()