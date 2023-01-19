#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:21:41 2023

@author: reka
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from pathlib import Path
import sys


U = float(sys.argv[2])
V = float(sys.argv[3])

ydata1 = np.loadtxt('optical_cond')
xdata1 = np.loadtxt('time')

xdata = xdata1[10:]
ydata = ydata1[10:]
  
def func(x, a, b):
    y = a + b*x 
    return y

parameters, covariance = curve_fit(func, xdata, ydata)
fit_a = parameters[0]
fit_b = parameters[1]

fit_cosine = func(xdata, fit_a, fit_b)
print('fit_a =', fit_a)
plt.plot(xdata1, ydata1, label='data')
plt.plot(xdata, fit_cosine, '-', label='fit')

'''
os.remove("optical_cond.txt")
os.remove("time.txt")
'''


Path("results").mkdir(parents=True, exist_ok=True)
os.chdir('results')
for i in range(0,len([name for name in os.listdir('.') if os.path.isfile(name)])+1):
    if os.path.exists('my_file'+str(i)) == True: 
        pass
    else: np.savetxt('my_file'+str(i), [fit_a, U, V])
    
#print(len([name for name in os.listdir('.') if os.path.isfile(name)]))
