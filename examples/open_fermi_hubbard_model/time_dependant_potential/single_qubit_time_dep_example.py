#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:03:50 2022

@author: reka
"""


"""
reproducing results section III (Fig.1) of following article: https://arxiv.org/abs/1811.05490
"""


import numpy as np 
import evos.src.lattice.spin_one_half_lattice as lat
import evos.src.methods.ed_time_dep_hamiltonian_lindblad_solver_new as solver
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

#number of sites
n_sites = 1
dim_H = 2 ** n_sites

# for lindblad solver
T = 30 # final time
dt = 0.01 #time step size
tsteps = int(T/dt)
t = np.linspace(0,T, tsteps)


spin_lat = lat.SpinOneHalfLattice(n_sites)


def H(t):   
    #omega_t = np.sqrt(2)*(1-np.cos(t))

    h = np.zeros((dim_H, dim_H), dtype = 'complex')
    h += -(1-np.cos(t))* spin_lat.sso('sz',0)

    return h



init_state = spin_lat.vacuum_state #vacuum state = all up
for i in np.arange(0,n_sites): #flip every second spin down
    init_state = np.dot( spin_lat.sso('sx',i), init_state.copy() )
    
init_state =1/2* np.array([[1,0], [0,1]], dtype = 'complex')

#init_state = np.array(init_state/LA.norm(init_state), dtype = 'complex')
print(init_state)



def L(t):  
    L = [0*spin_lat.sso('sz', 0), np.sqrt(3-0.5*np.sin(t))* spin_lat.sso('sm', 0) ,np.sqrt(2+ 0.5*np.sin(t))* spin_lat.sso('sp',0)]
    return L

def alpha(t):
    alpha = [0, 3-0.5*np.sin(t), 2+ 0.5*np.sin(t)]
    return(alpha)

# lindblad equation and solve
exp_sz , t11  = solver.SolveLindbladEquation(dim_H, H, L, dt, T).solve(spin_lat.sso('sz',0), init_state)


f,ax=plt.subplots(1)
#x=linspace(0,3*pi,1001)
#y=sin(x)
ax.plot(t/np.pi, exp_sz)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(plt.MultipleLocator(base=1.0))



