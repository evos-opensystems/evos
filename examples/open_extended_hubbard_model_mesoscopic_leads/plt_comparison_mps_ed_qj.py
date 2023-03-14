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



os.chdir('/home/reka/Dokumente/Masterarbeit_Anfang/mesoleads_topological_superconductivity/open_extend_hubb_mesoleads/1traj_mps_qj_compare_5')
b = np.loadtxt('n_av')
print(b[0])


os.chdir('/home/reka/Dokumente/Masterarbeit_Anfang/evos/examples/open_extended_hubbard_model_mesoscopic_leads/data_qj_seed1_1')
a = np.loadtxt('n_av_qj_ed_av')
#print(a)

T = []
for i in range(len(a)):
    T.append(i/1000)

T1 = []
for i in range(len(b[0])):
    T1.append(i/1000)


#print(T)
colorvector = ['#8c6bb1', '#fec44f', '#005a32', '#c7e9b4', '#e7298a', 'red' , 'green', 'blue']

plt.plot(T ,a , linestyle = 'dashed', color = colorvector[1])
plt.plot(T1 ,b[0] , linestyle = 'dashed', color = colorvector[0])
'''
plt.figure()
for i in range(0,8):
    plt.plot(T ,a[i], label = i, linestyle = 'dashed', color = colorvector[i])
    #plt.plot(T ,b1[i], label = i, linestyle = 'dashed', color = colorvector[i])
    plt.legend()


    
plt.show()
'''