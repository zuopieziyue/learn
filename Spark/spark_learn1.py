from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)



inputRDD = sc.textFile("log.txt")
errorsRDD = inputRDD.filter(lambda x: "error" in x)
warningsRDD = inputRDD.filter(lambda x: "waring" in x)
badlinesRDD = errorsRDD.union(warningsRDD)

print "Input had" + badlinesRDD.count() + "concerning lines"
print "Here are 10 examples:"

for line in badlinesRDD.take(10)
	print line




SparkContext.stop()