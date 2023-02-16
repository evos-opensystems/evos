#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:56:18 2023

@author: reka
"""

import numpy as np
import matplotlib.pyplot as plt 
import os




plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 12}) 
plt.rcParams["figure.figsize"] = (20,8)


a = np.loadtxt('n_av')
print(a)


T = []
for t in range(len(a[0])):
    T.append(t/100)
    
plt.plot(T,a[4])
