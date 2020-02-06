# encoding = 'utf-8'

import sys

sys.path.append("/home/hadoop/app/spark-2.3.0-bin-2.6.0-cdh5.7.0/python/")

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage spark0902.py <directory>", files=sys.stderr)
        sys.exit(-1)

    sc = SparkContext(appName="spark0902")
    ssc = StreamingContext(sc, 5)

    lines = ssc.textFileStream(sys.argv[1])
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()

    ssc.start()
