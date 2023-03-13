#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:03:50 2022

@author: reka
"""

import numpy as np 
import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice
import evos.src.methods.ed_time_dep_hamiltonian_lindblad_solver_new as solver
import matplotlib.pyplot as plt

# Open Fermi Hubbard Model with a time dependant Hamiltonian (sudden quench)

#number of sites
n_sites = 2
dim_H = 4 ** n_sites

# coupling parameters
t_hop = 1
U = 1
# oscillation frequency
omega = 0.11

# coupling of lindblad operators
alpha = 1

# for lindblad solver
T = 5 # final time
dt = 0.05 #time step size
tsteps = int(T/dt)
t = np.linspace(0,T, tsteps)
#print(t)
#print(tsteps)


spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_sites)

# Hamiltonian (can be time dependant)
def H(t): 
        
    hop = np.zeros((dim_H, dim_H), dtype = 'complex')
    for k in range(1, n_sites): 
        
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k+1,N), c_down_dag(k,N))
        hop += np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down'))
     
 
    coul = np.zeros((dim_H, dim_H), dtype = 'complex')
    for k in range(1, n_sites+1): 
        n_up =  np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
          
        coul += np.dot(n_up,n_down)

    
   
       
        
    
    H = -t_hop*hop +U*coul
   
    return H


# alternate spin up down state: first site: up, second site: down, third site: up and so on... 
def vac_ket(n_sites):
            
    vac_ket = np.zeros((dim_H, 1), dtype = 'complex')
    for i in range(0,dim_H+1):
        if i == 0:
            vac_ket[i,0] = 1
    
    return vac_ket

updown_ket = vac_ket(n_sites)

for i in np.arange(2,n_sites+1,2):
    
    updown_ket = np.dot(spin_lat.sso('adag',i-1, 'up'), updown_ket)
    updown_ket = np.dot(spin_lat.sso('adag',i, 'down'), updown_ket)


# Lindblad operators
def L(k, N, t):
    n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))#*np.exp(t)
    n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))#*np.exp(t)
    
    L = alpha*(n_up + n_down) 
    return L

L_list_time = []
for time in t:
    L_list = []
    for k in range(0, n_sites):
        L_list.append(L(k+1, n_sites, time))
    L_list_time.append(L_list)
print(np.shape(L_list_time))


# observable: number of up spins on first site
n_up_1 = np.dot(spin_lat.sso('adag',1, 'up'), spin_lat.sso('a',1, 'up'))

# right side of lindblad equation
#equation = solver.LindbladEquation(dim_H, H, L_list_left)

# solve lindblad equation
exp_n , t11  = solver.SolveLindbladEquation(dim_H, H, L, dt, T).solve(n_up_1, updown_ket)


#equation = solve.LindbladEquation(dim_H, H, L_list_left)
# solve lindblad eqaution and compute observable
#exp_n , t11  = solver.SolveLindbladEquation(dim_H, H, L_list, dt, T).solve(n_up_1, updown_ket)


plt.plot(t11, exp_n, label='$< \hat n_{up, 1}> $')
plt.title('Open Fermi-Hubbard Model with time dependant Hamiltonian')
plt.xlabel('t')
plt.ylabel('$< \hat n >$')
plt.legend()
plt.show()

