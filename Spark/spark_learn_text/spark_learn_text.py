#读入文件
textFile = spark.read.text("D:\\github\\learn\\Spark\\spark_learn_text\\namespace.txt")

#统计和显示
textFile.count()
textFile.first()
textFile.show()

#查询关键字
linesWithSpark = textFile.filter(textFile.value.contains("数据"))
textFile.filter(textFile.value.contains("数据")).count()

#持久化
linesWithSpark.cache()

#导入包
from pyspark.sql.functions import *


wordCounts = textFile.select(explode(split(textFile.value, "\s+")).alias("word")).groupBy("word").count()