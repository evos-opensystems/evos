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
'''
xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
print(Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')




'''
opt_cond = []
U = []
V = []
os.chdir('results') 
for filename in os.listdir(os.getcwd()):
    opt_cond.append(np.loadtxt(filename)[0])
    U.append(np.loadtxt(filename)[1])
    V.append(np.loadtxt(filename)[2])
    #print(np.loadtxt(filename))
    #with open(os.path.join(os.getcwd(results), filename), 'r') as f:
'''
opt_cond = np.array(opt_cond).reshape((len(U), len(V)))
u,v = np.meshgrid(U,V)
fig, ax = plt.subplots(1,1)
cp = ax.contourf(u,v,opt_cond)
ax.set_xlabel('U')
ax.set_ylabel('V')
plt.show()





# Arrays x, y and z for data plot visualization
x = np.arange(0, 25, 1)
y = np.arange(0, 25, 1)
# meshgrid makes a retangular grid out of two 1-D arrays. 
x, y = np.meshgrid(x, y)
z = x**2 + y**2  # x^2+y^2 

'''



  
# surface plot for x^2 + y^2 
fig = plt.figure() # creates space for a figure to be drawn 



# Uses a 3d prjection as model is supposed to be 3D
#axes = fig.gca(projection ='3d')
#fig.colorbar(axes)
# Plots the three dimensional data consisting of x, y and z 
sc = plt.scatter(U,V,c =opt_cond, s = 10,cmap='coolwarm' )
plt.colorbar(sc)
plt.xlabel('U')
plt.ylabel('V')


x1 = []
y1 = []
y2 = []
for x in np.linspace(0, 2.5):
    y = 0.5*x
    b = -0.5*x
    x1.append(x)
    y1.append(y)
    y2.append(b)
    
plt.plot(x1,y1, '--')
plt.plot(x1,y2, '--')
#axes.plot_surface(np.array(U), np.array(V), np.array(opt_cond)) 

# show command is used to visualize data plot   
plt.show() 
