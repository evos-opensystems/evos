"""
Created on Tue Dec 20 10:36:02 2022

@author: reka
"""

import evos
import evos.src.lattice.lattice as lat
import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice
#import evos.src.methods.lindblad as lindblad
import evos.src.methods.ed_quantum_jumps as ed_quantum_jumps
import evos.src.observables.observables as observables
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import scipy.linalg as la
import math
from numpy import linalg as LA


time_start = time.process_time()
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 11})



#parameters
J = 1
t = 1
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


gamma = 0
W = 10
seed_W = 1
rng = np.random.default_rng(seed=seed_W) # random numbers
#eps_vec = rng.uniform(0, W, n_sites) #onsite disordered energy random numbers
dt = 0.001
t_max = 5
n_timesteps = int(t_max/dt)
n_trajectories = 1
trajectory = 0 

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

#lattice
time_lat = time.process_time()
spin_lat = lat.Lattice('ed')
#spin_lat.specify_lattice('spin_one_half_lattice')
#spin_lat = spin_lat.spin_one_half_lattice.SpinOneHalfLattice(n_sites)
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

H = np.array(H_sys(t,J, V) + H_leads_left(eps_vector_l, k_vector_l, mu_L) + H_leads_right(eps_vector_r, k_vector_r, mu_R), dtype = 'complex')

print(H)
#np.save( 'time_H_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_H-time.process_time())
print('time_H:{0}'.format( time.process_time()- time_H ) )

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
        tot_init_state_ket = np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) # + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))

    if i > n_lead_left and i <= n_tot-n_lead_right:
        tot_init_state_ket = tot_init_state_ket
    
    if i > n_tot - n_lead_right:
        tot_init_state_ket = np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) #+ np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))

        



tot_init_state_ket_norm = tot_init_state_ket/LA.norm(tot_init_state_ket)

init_state = tot_init_state_ket_norm
    
print(init_state)


# print('LA.norm(init_state) :', la.norm(init_state))

#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n_av_qj_ed',(1,), n_timesteps) #1D
#obsdict.initialize_observable('sz_1',(1,), n_timesteps) #1D

n_av_qj_ed = np.dot(spin_lat.sso('adag',1, 'up'), spin_lat.sso('a',1, 'up'))
#print(sz_0)
#sz_1 = spin_lat.sso( 'sz', 1 )
###
# sz_0_init_state = np.dot( np.conjugate(init_state), np.dot(sz_0,init_state ))
# print(sz_0_init_state)
###

def compute_n_av_qj_ed(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.dot( np.dot( np.conjugate(state.T), n_av_qj_ed ), state )  )  
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('n_av_qj_ed',compute_n_av_qj_ed )
#obsdict.add_observable_computing_function('sz_1',compute_sz_1 )


#Lindbladian: dissipation only on central site
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

# L = np.array(L_list, dtype = 'complex')

    
ed_quantum_jumps = ed_quantum_jumps.EdQuantumJumps(n_tot, H, L_list)

#compute qj trajectories sequentially
for trajectory in range(n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = ed_quantum_jumps.quantum_jump_single_trajectory_time_evolution(init_state, t_max, dt, trajectory, obsdict )

#averages and errors
read_directory = os.getcwd()
write_directory = os.getcwd()


obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 


print('process time: ', time.process_time() - time_start )

#PLOT
n_av_qj_ed = np.loadtxt('n_av_qj_ed_av')
time_v = np.linspace(0, t_max, n_timesteps + 1  )
plt.plot(time_v, n_av_qj_ed, label= 'n_av_qj_ed', color = '#c7e9b4')
plt.legend()
plt.show()

