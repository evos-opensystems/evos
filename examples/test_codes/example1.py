#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:51:52 2023

@author: reka
"""

import numpy as np 
import sys 

t = float(sys.argv[1])
#k = 1
U = float(sys.argv[2])
V = float(sys.argv[3])



#print('example1: ', t)
print('example1: ', U)
#print('example1: ', V)
np.savetxt('example1_file', [t, U])