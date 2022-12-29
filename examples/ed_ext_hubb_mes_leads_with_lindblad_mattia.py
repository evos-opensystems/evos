#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

import math
from numpy import linalg as LA


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 11})


#DO BENCHMARK OF TEVO AND OBSERVABLES!
#parameters
J = 1
U = 1
V = 1
eps = 1
kappa = 1

alpha = 1

n_sites = 2 # number of system sites
n_lead_left = 1 # number of lindblad operators acting on leftest site
n_lead_right = 1 # number of lindblad operators acting on rightmost site

n_tot = n_sites + n_lead_left + n_lead_right

dim_H_sys = 4 ** n_sites

dim_H_lead_left =  4** n_lead_left 
dim_H_lead_right =  4** n_lead_right

dim_tot = dim_H_sys*dim_H_lead_left*dim_H_lead_right


# temperature and chemical potential on the different leads
T_L = 1
T_R = 1
mu_L = 1
mu_R = -1

####################################################################################################################
#fermi dirac distribution function
def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

# spectral function
def const_spec_funct(G,W,eps):
    if eps >= -W and eps <= W:
        return G
    else:
        return 0

G = 1
W= 1
#eps_step = 2*W/n_lead_left

#LEFT LEAD COEEFICIENTS
eps_step_l = 2* W / n_lead_left 
eps_vector_l = np.arange( -W, W, eps_step_l )
eps_delta_vector_l = eps_step_l * np.ones( len(eps_vector_l) )

k_vector_l = np.zeros( len(eps_vector_l) )
for i in range( len(eps_vector_l) ):
    k_vector_l[i] = np.sqrt( const_spec_funct( G , W, eps_vector_l[i] ) * eps_delta_vector_l[i]/ (2*math.pi) )  
    
#RIGHT LEAD COEEFICIENTS
eps_step_r = 2* W / n_lead_right
eps_vector_r = np.arange( -W, W, eps_step_r )
eps_delta_vector_r = eps_step_r * np.ones( len(eps_vector_r) )

k_vector_r = np.zeros( len(eps_vector_r) )
for i in range( len(eps_vector_r) ):
    k_vector_r[i] = np.sqrt( const_spec_funct( G , W, eps_vector_r[i] ) * eps_delta_vector_r[i]/ (2*math.pi) ) 
    


# Lindbladian parameers
alpha = 1



dt = 0.01
t_max = 10
n_timesteps = int(t_max/dt)

#os.chdir('benchmark')
#lattice
time_lat = time.process_time()
#spin_lat = lat.Lattice('ed')
#spin_lat.specify_lattice('spin_one_half_lattice')
spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_tot)
#np.save( 'time_lat_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lat-time.process_time())
print('time_lat:{0}'.format( time.process_time() - time_lat ) )

#Hamiltonian
time_H = time.process_time()

def H_sys(J, U, V): 
    
    # SYSTEM SITES 
    hop = np.zeros((dim_tot, dim_tot))
    for k in range(n_lead_left +1, n_tot - n_lead_right): 
        print('hopping terms on sites:', k, k+1)
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k+1,N), c_down_dag(k,N))
        hop += np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down'))
     
 
    coul = np.zeros((dim_tot, dim_tot))
    for k in range(n_lead_left +1, n_tot - n_lead_right +1): 
        print('coulomb terms on sites:', k)
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
          
        coul += np.dot(n_up,n_down)
        
    
    
    coul_nn = np.zeros((dim_tot, dim_tot))
    for k in range(n_lead_left +1, n_tot - n_lead_right):
        print('coulomb neighbour terms on sites:', k, k+1)
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
        
        n_up_nn = np.dot(spin_lat.sso('adag',k+1, 'up'), spin_lat.sso('a',k+1, 'up'))
        n_down_nn = np.dot(spin_lat.sso('adag',k+1, 'down'), spin_lat.sso('a',k+1, 'down'))
        
        
        coul_nn += np.dot(n_up, n_up_nn) + np.dot(n_up, n_down_nn) + np.dot(n_down, n_up_nn) + np.dot(n_down, n_down_nn)
    
    H = - J*hop + U* coul + V*coul_nn 
    return H


def H_leads_left(eps, k_vec, mu_L):
    # LEAD SITES - kinetic energy of left leads
    kin_leads = np.zeros((dim_tot, dim_tot))
    if n_lead_left == 0: 
        print('Ekin_lead left terms on sites:', 0)
        for k in range(0, dim_tot):
            kin_leads = np.zeros((dim_tot, dim_tot))
    else: 
        for k in range(1, n_lead_left+1): 
            print('Ekin_lead left terms on sites:', k)
            kin_leads += (eps[k-1] - mu_L) *( np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up')) + np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down')))
     
    
    
    # HOPPING BETWEEN LEADS AND SYSTEM LEFT SIDE
    hop_sys_lead = np.zeros((dim_tot, dim_tot))
    if n_lead_left == 0: 
        print('left sys lead hopping on sites:', 0)
        for k in range(0, dim_tot): 
            hop_sys_lead = np.zeros((dim_tot, dim_tot))
    else: 
        for k in range(n_lead_left, n_lead_left+1): 
            print('left sys lead hopping on sites:', k, k+1)
            hop_sys_lead += k_vec[k-1]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
     
        
    H = kin_leads + hop_sys_lead    
    return H

def H_leads_right(eps,k_vec, mu_R):
     # LEAD SITES - kinetic energy of left leads
    kin_leads = np.zeros((dim_tot, dim_tot))
    if n_lead_right == 0: 
        print('Ekin_lead right terms on sites:', 0)
        for k in range(0, dim_tot): 
            kin_leads = np.zeros((dim_tot, dim_tot))
    else:       
        for k in range(n_tot - n_lead_right + 1, n_tot+1):    
            print('Ekin_lead right terms on sites:', k)
            kin_leads += (eps[k - (n_tot - n_lead_right +1)] - mu_R) *( np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up')) + np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down')))
     
    # HOPPING BETWEEN LEADS AND SYSTEM RIGHT SIDE
    hop_sys_lead = np.zeros((dim_tot, dim_tot))
    if n_lead_right == 0: 
        print('right sys lead hopping on sites:', 0)
        for k in range(0, dim_tot):
            hop_sys_lead = np.zeros((dim_tot, dim_tot))
    else: 
        for k in range(n_tot - n_lead_right , n_tot): 
            print('right sys lead hopping on sites:', k, k+1)
            hop_sys_lead += k_vec[k - (n_tot - n_lead_right) ]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
    H = kin_leads + hop_sys_lead       
    return H

H = H_sys(U,J, V) + H_leads_left(eps_vector_l, k_vector_l, mu_L) + H_leads_right(eps_vector_r, k_vector_r, mu_R)

#print(H)

print('time_H:{0}'.format( time.process_time()- time_H ) )



# PREPARE SYSTEM SITES IN ANY CONVENIENT STATE

# no occupation of sites:
def vac():
            
    state_ket = np.zeros((dim_tot, 1))
    for i in range(0,dim_tot+1):
        if i == 0:
            state_ket[i,0] = 1

    return state_ket


#print(dim_tot)



tot_init_state_ket = vac()
for i in range(1, n_tot+1): 
    if i <= n_lead_left: 
        tot_init_state_ket = 1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))

    if i > n_lead_left and i <= n_tot-n_lead_right:
        tot_init_state_ket = tot_init_state_ket
    
    if i > n_tot - n_lead_right:
        tot_init_state_ket = 1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))

        



tot_init_state_ket_norm = tot_init_state_ket/LA.norm(tot_init_state_ket)

init_state = tot_init_state_ket_norm

#Lindbladian: 
def L_op(k, N):
    n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
    n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
    
    L = alpha*(n_up + n_down) 
    return L


#print(L(2,3))

L_list = []

for k in range(1, n_lead_left+1):
    print('k_left = ', k)
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* np.exp( 1/T_L * (eps_vector_l[k-1] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[k-1], mu_L))* spin_lat.sso('a',k, 'up'))
    #print(spin_lat.sso('a', k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* np.exp( 1/T_L * (eps_vector_l[k-1] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[k-1], mu_L)) * spin_lat.sso('a',k, 'down'))
    
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* fermi_dist(1/T_L, eps_vector_l[k-1], mu_L)) * spin_lat.sso('adag',k, 'up'))
    #print(spin_lat.sso('a', k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* fermi_dist(1/T_L, eps_vector_l[k-1], mu_L)) * spin_lat.sso('adag',k, 'down'))
    
  
    
for k in range(n_tot-n_lead_right +1, n_tot+1):
    print('k_right = ', k)
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* np.exp( 1/T_R * (eps_vector_r[k-(n_tot-n_lead_right +1)] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R))* spin_lat.sso('a',k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* np.exp( 1/T_R * (eps_vector_r[k-(n_tot-n_lead_right +1)] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R))* spin_lat.sso('a',k, 'down'))
    
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R)) * spin_lat.sso('adag',k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R)) * spin_lat.sso('adag',k, 'down'))
    
#L = gamma * np.matrix( spin_lat.sso( 'sm', 0 ) )  # int( n_sites/2 ) #Lindblad operators must be cast from arrays to matrices in order to be able to use .H
time_lind_evo = time.process_time()

L = L_list 

lindblad = lindblad.Lindblad(L ,H ,n_sites)


rho_0 = lindblad.ket_to_projector(init_state)        
rho_t = lindblad.solve_lindblad_equation(rho_0, dt, t_max)
#np.save( 'time_lind_evo_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lind_evo-time.process_time())
print('time_lind_evo:{0}'.format( time.process_time() - time_lind_evo ) )
#observables
time_lind_obs = time.process_time()

names_and_operators_list = {} #{'sz_0': spin_lat.sso('sz',0), 'sz_1': spin_lat.sso('sz',1), 'sz_2': , 'sz_3': sz_3 }
for i in range(1, n_tot+1):
    names_and_operators_list.update({'a_'+str(i) : np.dot(spin_lat.sso('adag',i, 'up'), spin_lat.sso('a',i, 'up')) })
obs_test_dict =  lindblad.compute_observables(rho_t, names_and_operators_list, dt, t_max )
#np.save( 'time_lind_obs_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lind_obs-time.process_time())
print('time_lind_obs:{0}'.format( time.process_time() - time_lind_evo ) )
#PLOT
time_v = np.linspace(0, t_max, n_timesteps )

for i in range(1, n_tot+1):
    plt.plot(time_v, obs_test_dict['a_'+str(i)], label = '<$\hat n$> on site '+str(i))    

plt.legend()
plt.xlabel('time')

beta_L = np.exp(- 1/T_L * (eps_vector_l[0] - mu_L) ) / ( np.exp(- 1/T_L * (eps_vector_l[0]-mu_L) ) + 1)

beta_list_L = []
for i in range(len(time_v)):
    beta_list_L.append(beta_L)
    
beta_R = np.exp( - 1/T_R * (eps_vector_r[0] - mu_R) ) / ( np.exp(- 1/T_R * (eps_vector_r[0]-mu_R) ) + 1)

beta_list_R = []
for i in range(len(time_v)):
    beta_list_R.append(beta_R)
    
    
plt.plot(time_v, beta_list_L, label = 'analytic thermalized expect. val', linestyle = 'dashed')    
plt.plot(time_v, beta_list_R, label = 'analytic thermalized expect. val', linestyle = 'dashed')    
plt.show()


# plt.imshow(sz_matrix, aspect='auto', extent=[0,t_max,1,n_sites])
# plt.yticks(range(1,n_sites+1))

# plt.colorbar()
# plt.xlabel(r'time $(1/\bar{J})$' )
# plt.ylabel('sites')
# plt.title('Lindblad evolution of the magnetization '+r'$\langle \hat{S_z} \rangle $ ' +'starting from the Neel state' +'\n Dissipation on central site c: ' + r'$\hat{L} = \gamma_c \hat{S}^-_c$ with $\gamma_c = 0.2 \bar{J}$. Disorder strength $W= 0.2 \bar{J}$ ')
# plt.show()    
