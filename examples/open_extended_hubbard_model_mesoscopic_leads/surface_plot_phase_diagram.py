#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:02:08 2023

@author: reka
"""

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import os
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 12}) 
plt.rcParams["figure.figsize"] = (20,8)



opt_cond = []
U = []
V = []
os.chdir('results2') 
for filename in os.listdir(os.getcwd()):
    opt_cond.append(np.loadtxt(filename)[0])
    U.append(np.loadtxt(filename)[1])
    V.append(np.loadtxt(filename)[2])

U1 = []
V1 = []
opt_cond1 = []
os.chdir('..') 
os.chdir('results3') 
for filename1 in os.listdir(os.getcwd()):
    opt_cond1.append(np.loadtxt(filename1)[0])
    U1.append(np.loadtxt(filename1)[1])
    V1.append(np.loadtxt(filename1)[2])
    #print(np.loadtxt(filename))
    #with open(os.path.join(os.getcwd(results), filename), 'r') as f:


x1 = []
y1 = []
y2 = []
for x in np.linspace(0, 5):
    y = 0.5*x
    b = -0.5*x
    x1.append(x)
    y1.append(y)
    y2.append(b)


fig, ax = plt.subplots()

ax1 = plt.subplot(121, xlim=[-5.5,8], ylim=[-5.5,6])
plt.plot(x1,y2, '--', c ='blue')
plt.plot(x1,y1, '--', c = 'blue')
plt.scatter(U1,V1,c = opt_cond1, s = 20,cmap='RdYlBu_r', vmin=0, vmax=0.02)
plt.xlabel('U/t')
plt.ylabel('V/t')
#plt.plot(x1,y2, '--')
ax1.annotate('$U = -2V$', xy = (2.3,-1) , c ='blue', textcoords="offset points", xytext=(0,10), ha='center')
#ax1.annotate('$U = 2V$', xy = (1.8,1) , c ='blue', textcoords="offset points", xytext=(0,10), ha='center')
textstr = '\n'.join((r'$\mathrm{T_{left}}=%.3f$' % (0.001, ),r'$\mathrm{T_{right}}=%.3f$' % (0.001, ), r'$\mathrm{N_{tot}}=%.0f$' % (4, ),  r'$\mathrm{N_{lead, L}}=%.0f$' % (1, ),  r'$\mathrm{N_{lead, R}}=%.0f$' % (1, ),  r'$\mu_L= %.0f$' % (1, ), r'$\mu_R=%.0f$' % (-1, )))
props = dict(boxstyle='square', facecolor= 'white', alpha=0.5)
# place a text box in upper left in axes coords
plt.text(0.9, 0.8, textstr, fontsize=13, verticalalignment='top', bbox=props, transform=plt.gcf().transFigure)


ax2 = plt.subplot(122, sharey=ax1, xlim=[-5.5,8], ylim=[-5.5,6])
im =plt.plot(x1,y2, '--', c = 'blue')
#plt.plot(x1,y1, '--', c = 'blue')
ax2.annotate('$U = -2V$', xy = (2.3,-1) , c ='blue', textcoords="offset points", xytext=(0,10), ha='center')
plt.scatter(U,V,c = opt_cond, s = 20,cmap='RdYlBu_r', vmin=0, vmax=0.02)
textstr = '\n'.join((r'$\mathrm{T_{left}}=%.3f$' % (1, ),r'$\mathrm{T_{right}}=%.3f$' % (1, ), r'$\mathrm{N_{tot}}=%.0f$' % (4, ),  r'$\mathrm{N_{lead, L}}=%.0f$' % (1, ),  r'$\mathrm{N_{lead, R}}=%.0f$' % (1, ),  r'$\mu_L= %.0f$' % (1, ), r'$\mu_R=%.0f$' % (-1, )))
props = dict(boxstyle='square', facecolor= 'white', alpha=0.5)
# place a text box in upper left in axes coords
plt.text(0.05, 0.8, textstr, fontsize=13, verticalalignment='top', bbox=props, transform=plt.gcf().transFigure)


plt.ylabel('V/t')
plt.xlabel('U/t')
cbar = plt.colorbar()
cbar.set_label('$\sigma$ (optical conductivity)')
#plt.xlim(-5,5)
fig.suptitle('Open Extended Hubbard Model in Mesoscopic Lead Bath at two different temperatures (T = 1 and T = 0.001)')
plt.show()


