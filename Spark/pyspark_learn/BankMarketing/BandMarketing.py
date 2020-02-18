# encoding = 'utf-8'

import sys

sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")


from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder


DATAPATH = "file:///home/hadoop/data/bankMarketing/bank_marketing_data.csv"

if __name__ == '__main__':
    # 实例化SparkSession对象
    spark = SparkSession.builder.appName("BankMarketing").getOrCreate()

    # 设置日志级别，减少日志输出，便于查看运行结果
    spark.sparkContext.setLogLevel("WARN")

    # 读取数据
    bank_Marketing_Data = spark.read\
        .format("csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .load(DATAPATH)

    bank_Marketing_Data.show(5)

    # 读取营销数据指定的11个字段，并将age、duration、previous
    # 三个字段的类型从Integer类型转换为Double类型
    selected_Data = bank_Marketing_Data.select(
        "age",
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "duration",
        "previous",
        "poutcome",
        "empvarrate",
        "y")\
        .withColumn("age", bank_Marketing_Data["age"].cast(DoubleType()))\
        .withColumn("duration", bank_Marketing_Data["duration"].cast(DoubleType()))\
        .withColumn("previous", bank_Marketing_Data["previous"].cast(DoubleType()))
    selected_Data.show(5)
    print(selected_Data.count())

    # 对数据进行概要统计
    summary = selected_Data.describe()
    print("Summary Statistics:")
    summary.show()

    # 查看每一列所包含的不同值数量
    columnNames = selected_Data.columns
    for field in columnNames:
        uniqueValues = field + "\t" + str(selected_Data.select(field).distinct().count())
        print(uniqueValues)

    # 特征工程
    indexer = StringIndexer(inputCol="job", outputCol="jobIndex")
    indexed = indexer.fit(selected_Data).transform(selected_Data)

    encoder = OneHotEncoder(dropLast=False, inputCol="jobIndex", outputCol="jobVec")
    encoded = encoder.transform(indexed)

    indexer = StringIndexer(inputCol="marital", outputCol="maritalIndex")
    indexed = indexer.fit(selected_Data).transform(selected_Data)












