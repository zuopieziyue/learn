# encoding = 'utf-8'

import sys

sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")

from pyspark.sql import SparkSession, types, functions
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

DATAPATH = "file:///home/hadoop/data/"

if __name__ == '__main__':
    spark = SparkSession.builder.appName("spark").getOrCreate()

    # 数据结构
    fieldSchema = StructType([StructField("TID", StringType(), True),
                              StructField("Lat", DoubleType(), True),
                              StructField("Lon", DoubleType(), True),
                              StructField("Time", StringType(), True)])

    # 读取数据
    taxiDF = spark.read.format("csv").option("header", "false") \
        .schema(fieldSchema) \
        .load(DATAPATH + "taxi.csv")

    # 设置参数
    va = VectorAssembler(
        inputCols=["Lat", "Lon"],
        outputCol="features"
    )

    # 将数据集按照指定的特征向量进行转化
    taxiDF2 = va.transform(taxiDF)

    # 数据持久化
    taxiDF2.cache()

    # 设置训练集和数据集的比例
    trainTestRatio = [0.7, 0.3]
    (trainingData, testData) = taxiDF2.randomSplit(trainTestRatio, 2333)

    # 设置模型的参数
    km = KMeans(
        k=10,
        featuresCol="features",
        predictionCol="prediction",
    )

    # 训练KMeans模型，此步骤比较耗时
    kmModel = km.fit(trainingData)
    kmResult = kmModel.clusterCenters()
    for result in kmResult:
        print(result)

    # 对测试数据进行聚类
    predictions = kmModel.transform(testData)
    predictions.show()

    # 将预测数据注册为临时表
    predictions.registerTempTable("perdictions")

    # 每天哪个时段的出租车最繁忙
    # 使用select方法选取字段，substring用于提取时间的前2为作为小时，
    # alias方法为选取的字段命名一个别名，
    # groupBy方法对结果进行分组
    tmpQuery = predictions.select(substring("Time", 0, 2).alias("hour"), "prediction") \
        .groupBy("hour", "prediction")

    # agg为聚集函数，count为其中一种实现，用于统计某个字段的数量
    # 最后的结果按照预测命中数来降序排列（Desc）
    predictCount = tmpQuery.agg({"prediction": "count"})\
        .withColumnRenamed("count(prediction)", "count")\
        .orderBy(desc("count"))
    predictCount.show()

    # 将统计的时段数据保存
    predictCount.rdd.saveAsTextFile(DATAPATH + "predictCount")

    # 哪个区域出租车最繁忙
    busyZones = predictions.groupBy("prediction").count()
    busyZones.show()

    # 将统计的地域数据保存
    busyZones.rdd.saveAsTextFile(DATAPATH + "busyZones")

    spark.stop()
