# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:41:27 2019

@author: gongyue
"""

import os

def get_item_info(input_file):
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    item_info = {}
    fp = open(input_file, encoding='utf-8')
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue

        item = line.strip().split(',')
        if len(item) < 3:
            continue
        elif len(item) == 3:
            itemid, title, genre = item[0], item[1], item[2]
        elif len(item) > 3:
            itemid = item[0]
            genre = item[-1]
            title = ','.join(item[1:-1])
        item_info[itemid] = [title, genre]
    fp.close()
    return item_info

def get_ave_score(input_file):
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    record_dict = {}
    fp = open(input_file, encoding="utf-8")
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        print (len(item))
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], item[2]
        if itemid not in record_dict:
            record_dict[itemid] = [0, 0]
        record_dict[itemid][0] += 1
        record_dict[itemid][1] += rating
    fp.close()
    for itemid in record_dict:
        score_dict[itemid] = rount(record_dict[itemid][1] / record_dict[itemid][0], 3)
    return score_dict

if __name__ == '__main__':
    #item_dict = get_item_info('data\\movies.txt')
    #print(len(item_dict))
    #print(item_dict["1"])
    #print(item_dict["11"])

    score_dict = get_ave_score("data\\ratings.txt")
    print(len(score_dict))


