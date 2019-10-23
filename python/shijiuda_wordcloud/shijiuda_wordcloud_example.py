# -*- coding: utf-8 -*-
import re
import jieba
import numpy
import pandas
import codecs

#匹配中文的分词
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

#w_18 = codecs.open('shibada.txt', 'r', 'utf8')
#w_18_Content = w_18.read()
#w_18.close()

w_19 = codecs.open('shijiuda.txt', 'r', 'utf8')
print w_19
w_19_Content = w_19.read()
w_19.close()

stat = []
stopwords = set([
    '的', '和', '是', '在', '要', 
    '为', '我们', '以', '把', '了',
    '到', '上', '有'
])

#segs = jieba.cut(w_18_Content)
#for seg in segs:
#    if zhPattern.search(seg):
#        if seg not in stopwords:
#            stat.append({
#                'word': seg,
#                'from': '十八大'
#            })

segs = jieba.cut(w_19_Content)
for seg in segs:
    if zhPattern.search(seg):
        if seg not in stopwords:
            stat.append({
                'word': seg,
                'from': '十九大'
            })

statDF = pandas.DataFrame(stat)

ptStat = statDF.pivot_table(
    index='word', 
    columns='from', 
    fill_value=0,
    aggfunc=numpy.size
)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(
    font_path='D:\\PDM\\2.4\\simhei.ttf', 
    background_color="black"
)

words = ptStat.set_index('segment').to_dict()

wordcloud.fit_words(words['计数'])

plt.imshow(wordcloud)
