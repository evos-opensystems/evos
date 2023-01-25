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

os.chdir('data_qj')
a = np.loadtxt('sz_0_av')
#print(a[0])


T = []
for t in range(len(a)):
    T.append(t)
    
plt.plot(T,a)
