#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:52:10 2023

@author: reka
"""
import numpy as np
import sys 


t = float(sys.argv[1])
U = float(sys.argv[2])
V = float(sys.argv[3])

#print('example2: ', t)
print('example2: ', U)
#print('example2: ', V)


xdata1 = np.loadtxt('example1_file')
print(xdata1)