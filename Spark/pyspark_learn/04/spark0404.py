# encoding = 'utf-8'

import os
import sys

os.environ['SPARK_HOME'] = '/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0'
sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")

from pyspark import SparkConf, SparkContext

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: topn <input>', file=sys.stderr)
        sys.exit(-1)

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    ageData = sc.textFile(sys.argv[1]).map(lambda x: x.split(' ')[1])
    totalAge = ageData.map(lambda age: int(age)).reduce(lambda a, b: a+b)
    counts = ageData.count()
    avgAge = totalAge/counts

    print(totalAge)
    print(counts)
    print(avgAge)

    sc.stop()