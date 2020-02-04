# encoding = 'utf-8'

import sys

sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")

from pyspark.sql import SparkSession
from pyspark import Row

def basic(spark):
    df = spark.read.json('file:///home/hadoop/script/myspark/08/data/0801/people.json')
    df.show()
    df.printSchema()
    df.select(df['name'], df['age']+1).show()
    df.filter(df['age'] > 25).show()
    df.createOrReplaceTempView("people")
    sqlDF = spark.sql("select * from people")
    sqlDF.show()

def schema_inference_eample(spark):
    sc = spark.sparkContext
    lines = sc.textFile("file:///home/hadoop/script/myspark/08/data/0801/people.txt")
    parts = lines.map(lambda l: l.split(","))
    people = parts.map(lambda p: Row(name=p[0], age=int(p[1])))

    schemaPeople = spark.createDataFrame(people)
    schemaPeople.createOrReplaceTempView("people")

    tenagers = spark.sql("select name from people where age >=13 and age <= 19")
    tenagers.show()


if __name__ == '__main__':
    spark = SparkSession.builder.appName('spark0801').getOrCreate()

    #basic(spark)
    schema_inference_eample(spark)

    spark.stop()
