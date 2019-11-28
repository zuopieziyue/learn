# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:26:49 2019

@author: gongyue
"""

import numpy as np

a = np.array([1,0,1])
b = np.array([1,1,0])

sum = 0
for i,j in zip(a,b):
	sum += i*j
print(sum)
print(a.dot(b))
	