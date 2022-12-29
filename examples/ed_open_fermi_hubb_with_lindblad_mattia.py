#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:36:02 2022

@author: reka
"""

import evos
#import evos.src.lattice as spin_lat
import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice
import evos.src.methods.lindblad as lindblad
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.linalg as la


#DO BENCHMARK OF TEVO AND OBSERVABLES!
#parameters
n_sites = 2
dim_H = 4 ** n_sites

#hamiltonian parameters
J = 1
U = 1

# Lindbladian parameers
alpha = 1


W = 10
seed = 1
np.random.seed(seed)
eps_vec = np.random.uniform(0, W, n_sites)
dt = 0.01
t_max = 10
n_timesteps = int(t_max/dt)


time_lat = time.process_time()

spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_sites)

print('time_lat:{0}'.format( time.process_time() - time_lat ) )

#Hamiltonian
time_H = time.process_time()

def H(J, U): 
        
    hop = np.zeros((dim_H, dim_H), dtype = 'complex')
    for k in range(1, n_sites): 
        
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k+1,N), c_down_dag(k,N))
        hop += np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down'))
     
 
    coul = np.zeros((dim_H, dim_H), dtype = 'complex')
    for k in range(1, n_sites+1): 
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
          
        coul += np.dot(n_up,n_down)
        
    
        
    H = - J* hop + U* coul
    return H

H = H(U,J)

print('time_H:{0}'.format( time.process_time()- time_H ) )



def vac_ket(n_sites):           
    state_ket = np.zeros((dim_H, 1))
    for i in range(0,dim_H+1):
        if i == 0:
            state_ket[i,0] = 1
    return state_ket


init_state = vac_ket(n_sites)
for i in np.arange(2,n_sites+1,2):
    #updown_ket = np.dot(c_up_dag(i-1, N), updown_ket)
    init_state = np.dot(spin_lat.sso('adag',i-1, 'up'), init_state)
    
    #updown_ket = np.dot(c_down_dag(i, N), updown_ket)
    init_state = np.dot(spin_lat.sso('adag',i, 'down'), init_state)

#Lindbladian: 
def L_op(k, N):
    n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
    n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
    
    L = alpha*(n_up + n_down) 
    return L




L = []
for k in range(0, n_sites):
    L.append(np.matrix(L_op(k+1, n_sites)))
    

time_lind_evo = time.process_time()
lindblad = lindblad.Lindblad(L ,H ,n_sites)


rho_0 = lindblad.ket_to_projector(init_state)        
rho_t = lindblad.solve_lindblad_equation(rho_0, dt, t_max)

print('time_lind_evo:{0}'.format( time.process_time() - time_lind_evo ) )


#observables
time_lind_obs = time.process_time()

names_and_operators_list = {} 
for i in range(1, n_sites+1):
    names_and_operators_list.update({'a_'+str(i) : np.dot(spin_lat.sso('adag',i, 'up'), spin_lat.sso('a',i, 'up')) })
obs_test_dict =  lindblad.compute_observables(rho_t, names_and_operators_list, dt, t_max )

print('time_lind_obs:{0}'.format( time.process_time() - time_lind_evo ) )
#PLOT
time_v = np.linspace(0, t_max, n_timesteps )

for i in range(1, n_sites+1):
    plt.plot(time_v, obs_test_dict['a_'+str(i)], label = '<a> on site '+str(i))    

plt.legend()
plt.xlabel('time')
plt.show()



