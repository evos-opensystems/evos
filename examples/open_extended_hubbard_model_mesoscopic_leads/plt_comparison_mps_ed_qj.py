#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:01:43 2023

@author: reka
"""



import numpy as np
import matplotlib.pyplot as plt 
import os




plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 12}) 
plt.rcParams["figure.figsize"] = (20,8)



os.chdir('/home/reka/Dokumente/Masterarbeit_Anfang/mesoleads_topological_superconductivity/open_extend_hubb_mesoleads/comparison_mps_qj/new_solver_test4')
b = np.loadtxt('n_av')
print(b[1])


os.chdir('/home/reka/Dokumente/Masterarbeit_Anfang/mesoleads_topological_superconductivity/open_extend_hubb_mesoleads/comparison_ed_qj/test_4')
a = np.loadtxt('n_av')
print(a[1])



T1 = []
for i in range(len(b[3])):
    T1.append(i/100)


#print(T)
colorvector = ['#8c6bb1', '#fec44f', '#005a32', '#c7e9b4', '#e7298a', 'red' , 'green', 'blue']

plt.plot(T1 ,a[0]   , linestyle = 'dashed', color = colorvector[2], label = 'ed qj')
plt.plot(T1 ,b[0] , linestyle = 'dashed', color = colorvector[1], label = 'mps qj')
plt.legend()
