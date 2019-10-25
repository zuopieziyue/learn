import sys
import jieba
import jieba.analyse
import pandas as pd

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

music_meta_file = "D:\\mygithub\\data\\learn\\cut\\music_data\\music_meta"


#读取文件
music_meta = pd.read_csv(music_meta_file,
						sep='\001',
						nrows = 10,
						names = ['item_id','item_name','item_name2','category','location','labels'])

del music_meta['item_name2']

#分词
#music_meta['cut_name'] = music_meta.item_name.apply(lambda x:' '.join(jieba.cut(x, cut_all=True)))

music_meta['cut_name1'] = music_meta.item_name.apply(lambda x:
	' '.join([str(x)+'_'+str(y) for x,y in jieba.analyse.extract_tags(x, topK=6, withWeight=True)]))
print(music_meta.head())










