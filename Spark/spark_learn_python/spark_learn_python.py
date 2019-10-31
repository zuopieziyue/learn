from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from pyspark.sql import Row



if __name__=="__main__":
	conf = SparkConf().setMaster("local").setAppName("My App")
	sc.stop()
	sc = SparkContext(conf = conf)
	
	lines = sc.textFile("namespace.txt")
	result = lines.take(20)
	
	
	
	
	
	print(result[1])
	#lines2 = lines.filter()
	
	
	#lineFirst = lines.first(10)
	
	#print(lineFirst)
	
	#count = lines.count()
	#print(count)

	
	
	
	
	
	
	