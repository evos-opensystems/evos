# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:02:08 2023

@author: reka
"""

import numpy as np
import matplotlib.pyplot as plt 
import os




plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 12}) 
plt.rcParams["figure.figsize"] = (20,8)



opt_cond = []
U = []

os.chdir('0') 
a = np.loadtxt('sz')
print(a[0])

os.chdir('..') 
os.chdir('1') 
b = np.loadtxt('sz')
print(b[0])
    
T = []
for t in range(len(a[0])):
    T.append(t)
    
plt.plot(T,a[0])
plt.plot(T,b[0])