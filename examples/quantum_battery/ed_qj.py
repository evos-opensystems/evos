"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with ed quantum jumps. The dimension of the oscillator needs to be strongly truncated.
"""

import evos.src.lattice.dot_with_oscillator_lattice as lattice 
import evos.src.methods.partial_traces.partial_trace_tls_boson as pt 
import evos.src.methods.ed_quantum_jumps as ed_quantum_jumps
import evos.src.observables.observables as observables

import numpy as np 
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as la
import sys
#import math
import os
np.set_printoptions(threshold=sys.maxsize)

#PARAMETERS
max_bosons = 2
eps = 1
Om_kl = 1
Om_kr = 1
g_kl = 1
g_kr = 1
om_0 = 1
F = 1
#FIXME: how do I compute N0??

Gamma = 1
mu_l = +1
mu_r = +1
T_l = 1
T_r = 1
k_b = 1 #boltzmann constant
 
dt = 0.05
t_max = 100
time_v = np.arange(0, t_max, dt)
n_timesteps = int(t_max/dt)
n_trajectories = 50
trajectory = 0

#LATTICE
lat = lattice.DotWithOscillatorLattice(max_bosons)

class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps):
        
        h_s = eps * lat.sso('ch',1) @ lat.sso('c',1)
        return h_s 

    def h_b(self, Om_kl, Om_kr, mu_l, mu_r):
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = ( Om_kl + mu_l ) * lat.sso('ch',0) @ lat.sso('c',0) + ( Om_kr + mu_r ) * lat.sso('ch',3) @ lat.sso('c',3)
        return h_b
   
    def h_t(self, g_kl, g_kr):
        h_t = g_kl * ( lat.sso('ch',1) @ lat.sso('c',0) + lat.sso('ch',0) @ lat.sso('c',1) ) + g_kr * ( lat.sso('ch',1) @ lat.sso('c',3) + lat.sso('ch',3) @ lat.sso('c',1) )
        return h_t
    
    def h_v(self, om_0, F):
        #FIXME: need to detract N0
        #FIXME: is m = 1 ?
        h_v = om_0 * lat.sso('ah',2) @ lat.sso('a',2) - F * lat.sso('ch',1) @ lat.sso('c',1) @ ( lat.sso('ah',2) + lat.sso('a',2) ) 
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, mu_l, mu_r, g_kl, g_kr, om_0, F):
        h_tot = self.h_s(eps) + self.h_b(Om_kl, Om_kr, mu_l, mu_r) + self.h_t(g_kl, g_kr) + self.h_v(om_0, F)
        return h_tot
        
    
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)
# h_s = ham.h_s(eps)
# h_b = ham.h_b(Om_kl, Om_kr, mu_l, mu_r)
# h_t = ham.h_t(g_kl, g_kr)
# h_v = ham.h_v(om_0, F)
h_tot = ham.h_tot(eps, Om_kl, Om_kr, mu_l, mu_r, g_kl, g_kr, om_0, F)


#Lindblad operators
def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

def lindblad_op_list_left_lead( Om_kl, Gamma, mu_l, T_l ):
    l_list_left = []
    l_list_left.append( np.sqrt( Gamma * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l ) ) * lat.sso( 'c',0 ) )
    l_list_left.append( np.sqrt( Gamma * fermi_dist( 1./T_l, Om_kl, mu_l)) * lat.sso('ch',0) )
    return l_list_left

def lindblad_op_list_right_lead( Om_kr, Gamma, mu_r, T_r ):
    l_list_right = []
    l_list_right.append( np.sqrt( Gamma * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) ) * lat.sso( 'c',3 ) )
    l_list_right.append( np.sqrt( Gamma * fermi_dist( 1./T_r, Om_kr, mu_r)) * lat.sso('ch',3) )
    return l_list_right

l_list_left = lindblad_op_list_left_lead( Om_kl, Gamma, mu_l, T_l )
l_list_right = lindblad_op_list_right_lead( Om_kr, Gamma, mu_r, T_r )
l_list = l_list_left + l_list_right

#Initial State: using vacuum for now
#NOTE: vacuum for leads (compare with ed qj) or thermal state on leads (compare with doubled qj?
init_state = lat.vacuum_state

#Observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n_system',(1,), n_timesteps) #1D
obsdict.initialize_observable('U',(1,), n_timesteps) #1D

def compute_n_system(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.conjugate(state) @ lat.sso('ch',1) @ lat.sso('c',1) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

def compute_U(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = om_0 * np.real( np.conjugate(state) @ lat.sso('ah',2) @ lat.sso('a',2) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('n_system',compute_n_system)
obsdict.add_observable_computing_function('U',compute_U)
#NOTE: von Neuman entropy is not computed here. It should be possible to compute it with syten

#compute QJ time evolution
os.chdir('data_qj_ed')
ed_quantum_jumps = ed_quantum_jumps.EdQuantumJumps(4, h_tot, l_list)
#compute qj trajectories sequentially
for trajectory in range(n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = ed_quantum_jumps.quantum_jump_single_trajectory_time_evolution(init_state, t_max, dt, trajectory, obsdict )

#averages and errors
read_directory = os.getcwd()
write_directory = os.getcwd()

obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 

#PLOT                                                                              
n_system_av = np.loadtxt('n_system_av')

plt.plot(time_v, n_system_av[:-1], label='n_system_av')

plt.legend()
plt.show()
