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
import shutil
import scipy.linalg as la
import evos.src.methods.ed_quantum_jumps as ed_quantum_jumps
import evos.src.observables.observables as observables

#os.chdir('benchmark')
try:
    os.system('mkdir data_qj')
    os.chdir('data_qj')
except:
    pass

try:
    shutil.rmtree('0')
    shutil.rmtree('1')
except:
    pass


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
n_trajectories = 1
trajectory = 0 

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
    state_ket = np.zeros((dim_H))
    for i in range(0,dim_H+1):
        if i == 0:
            state_ket[i] = 1
    return state_ket


init_state = vac_ket(n_sites)
for i in np.arange(2,n_sites+1,2):
    #updown_ket = np.dot(c_up_dag(i-1, N), updown_ket)
    init_state = np.dot(spin_lat.sso('adag',i-1, 'up'), init_state)
    
    #updown_ket = np.dot(c_down_dag(i, N), updown_ket)
    init_state = np.dot(spin_lat.sso('adag',i, 'down'), init_state)
    

print(np.array(init_state))


#observables
time_lind_obs = time.process_time()


#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n_up',(1,), n_timesteps) #1D


n_up = np.dot(spin_lat.sso('adag',1, 'up'), spin_lat.sso('a',1, 'up'))
print(np.shape(n_up))
print(np.shape(init_state))

def compute_n_up(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.dot( np.dot( np.conjugate(state),n_up), state )  )  
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('n_up', compute_n_up )


#Lindbladian: 
def L_op(k, N):
    n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
    n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
    
    L = alpha*(n_up + n_down) 
    return L

L = []
for k in range(0, n_sites):
    L.append(np.matrix(L_op(k+1, n_sites)))



ed_quantum_jumps = ed_quantum_jumps.EdQuantumJumps(n_sites, H, L)

for trajectory in range(n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = ed_quantum_jumps.quantum_jump_single_trajectory_time_evolution(init_state, t_max, dt, trajectory, obsdict )

#averages and errors
read_directory = os.getcwd()
write_directory = os.getcwd()


obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 


#print('process time: ', time.process_time() - time_start )

#PLOT
exp_n_up = np.loadtxt('n_up_av')
time_v = np.linspace(0, t_max, n_timesteps + 1  )
plt.plot(time_v,exp_n_up, label= 'n_up_av')
plt.legend()
plt.show()

