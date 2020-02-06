# encoding = 'utf-8'

import sys

sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: spark0901.py <hostname> <port>', file=sys.stderr)
        sys.exit(-1)

    sc = SparkContext(appName="spark0901")
    ssc = StreamingContext(sc, 5)

    # TODO 根据业务需求开发我们自己的业务
    lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a+b)

    counts.pprint()

    ssc.start()

    ssc.awaitTermination()