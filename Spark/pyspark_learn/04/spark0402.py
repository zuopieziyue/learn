# encoding = 'utf-8'

import os
import sys

os.environ['SPARK_HOME'] = '/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0'
sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: wordcount <input> <output>', file=sys.stderr)
        sys.exit(-1)

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    def printResult():
        counts = sc.textFile(sys.argv[1]) \
            .flatMap(lambda line: line.split('\t')) \
            .map(lambda x: (x, 1)) \
            .reduceByKey(lambda a, b: a+b)

        output = counts.collect()

        for (word, count) in output:
            print('%s: %i' % (word, count))

    def saveFile():
        counts = sc.textFile(sys.argv[1]) \
            .flatMap(lambda line: line.split('\t')) \
            .map(lambda x: (x, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .saveAsTextFile(sys.argv[2])

    saveFile()

    sc.stop()
