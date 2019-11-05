# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:29:22 2019

@author: Administrator
"""
import os
import pandas as pd

data_path = 'D:\\mygithub\\data\\learn\\python\\recsys_music\\raw_data\\'

'''
data file
'''
music_meta = os.path.join(data_path,'music_meta')
user_profile = os.path.join(data_path, 'user_profile.data')
user_watch_pref = os.path.join(data_path, 'user_watch_pref.sml')

# 相似度矩阵的存储
sim_mid_data_path = ''

'''
load raw data format
'''
#item description data
def gen_music_meta():
	df_music_meta = pd.read_csv(music_meta,
						sep='\001',
						names=['item_id','item_name','desc','total_timelen','location','tags'])
	del df_music_meta['desc']
	return df_music_meta.fillna('-')

# user profile data
def gen_user_profile():
	return pd.read_csv(user_profile,
						sep=',',
						#nrows=10,
						names=['user_id','gender','age','salary','provice'])

# user action data
def gen_user_watch():
	return pd.read_csv(user_profile,
						sep='\001',
						names=['user_id','item_id','stay_seconds','hour'])

