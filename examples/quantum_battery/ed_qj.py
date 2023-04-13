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
sys.stdout.write('test')
import argparse

arg_parser = argparse.ArgumentParser(description = "Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with ed lindblad. The dimension of the oscillator needs to be strongly truncated.")
arg_parser.add_argument("-b",   "--bosons", dest = 'max_bosons',  default = 4, type = int, help = 'number of bosonic degrees of freedom - 1 [4]')
arg_parser.add_argument("-dt",   "--timestep", dest = 'dt',  default = 0.02, type = float, help = 'timestep [0.02]')
arg_parser.add_argument("-t_max",   "--max_time", dest = 't_max',  default = 5, type = float, help = 'maximal simulated time [5]')
arg_parser.add_argument("-mu_l",   "--checmical_pot_left_lead", dest = 'mu_l',  default = +0.5, type = float, help = 'checmical pot. left lead [0.5]')
arg_parser.add_argument("-mu_r",   "--checmical_pot_right_lead", dest = 'mu_r',  default = -0.5, type = float, help = 'checmical pot. right lead [-0.5]')

#FIXME: ADD MU_L AND MU_R
args = arg_parser.parse_args()


np.set_printoptions(threshold=sys.maxsize)
sys.stdout.write('test')

#PARAMETERS
max_bosons = args.max_bosons

om_0 = 0.2
m = 1
lamb = 0.1
x0 = np.sqrt( 2./ (m * om_0) )
F = 2 *lamb / x0

eps = 0  
Om_kl = +0.5
Om_kr = -0.5
Gamma = 2
g_kl = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
g_kr = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
N0 = 0.5 #FIXME: is this correct?
delta_l = 1
delta_r = 1

mu_l = args.mu_l
mu_r = args.mu_r

T_l = 1./0.5 #beta_l = 0.5
T_r = 1./0.5 #beta_r = 0.5
k_b = 1 #boltzmann constant
 
dt = args.dt
t_max = args.t_max
time_v = np.arange(0, t_max, dt)
n_timesteps = int(t_max/dt)
n_trajectories = 1
first_trajectory = 0

#LATTICE
lat = lattice.DotWithOscillatorLattice(max_bosons)

class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps): #system
        h_s = eps * lat.sso('ch',1) @ lat.sso('c',1)
        return h_s 

    def h_b(self, Om_kl, Om_kr): #leads
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = Om_kl * lat.sso('ch',0) @ lat.sso('c',0) + Om_kr * lat.sso('ch',3) @ lat.sso('c',3)
        return h_b
   
    def h_t(self, g_kl, g_kr): #system-leads
        h_t = g_kl * ( lat.sso('ch',1) @ lat.sso('c',0) + lat.sso('ch',0) @ lat.sso('c',1) ) + g_kr * ( lat.sso('ch',1) @ lat.sso('c',3) + lat.sso('ch',3) @ lat.sso('c',1) )
        return h_t
    
    def h_boson(self, om_0): #oscillator
        h_boson = om_0 * lat.sso('ah',2) @ lat.sso('a',2)
        return h_boson
    
    def h_v(self, F): #system-oscillator
        dimH = lat.sso('c',0).shape[0]
        h_v = - F * ( lat.sso('ch',1) @ lat.sso('c',1) - N0 * np.eye(dimH, dtype='complex') ) @ ( lat.sso('ah',2) + lat.sso('a',2) ) 
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F):
        h_tot = + self.h_boson(om_0) + self.h_v(F) +self.h_s(eps) + self.h_t(g_kl, g_kr) + self.h_b(Om_kl, Om_kr)
        return h_tot
        
 
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)
# h_s = ham.h_s(eps)
# h_b = ham.h_b(Om_kl, Om_kr, mu_l, mu_r)
# h_t = ham.h_t(g_kl, g_kr)
# h_v = ham.h_v(om_0, F)
h_tot = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F)
# print('h_tot is symmetric: ', ( h_tot == h_tot.T ).all() )
# quit()

#Lindblad operators
def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

def lindblad_op_list_left_lead( Om_kl, delta_l, mu_l, T_l ):
    l_list_left = []
    l_list_left.append( np.sqrt( delta_l * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l ) ) * lat.sso( 'c',0 ) )
    l_list_left.append( np.sqrt( delta_l * fermi_dist( 1./T_l, Om_kl, mu_l)) * lat.sso('ch',0) )
    return l_list_left

def lindblad_op_list_right_lead( Om_kr, delta_r, mu_r, T_r ):
    l_list_right = []
    l_list_right.append( np.sqrt( delta_r * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) ) * lat.sso( 'c',3 ) )
    l_list_right.append( np.sqrt( delta_r * fermi_dist( 1./T_r, Om_kr, mu_r)) * lat.sso('ch',3) )
    return l_list_right

l_list_left = lindblad_op_list_left_lead( Om_kl, delta_l, mu_l, T_l )
l_list_right = lindblad_op_list_right_lead( Om_kr, delta_r, mu_r, T_r )
l_list = l_list_left + l_list_right

#Initial State: using vacuum for now
#NOTE: vacuum for leads (compare with ed qj) or thermal state on leads (compare with doubled qj?
init_state = lat.vacuum_state

# FIXME exite one particle in the left lead: USED TO DEBUGG
init_state = lat.sso('ch',0) @ init_state

#Observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n_system',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_3',(1,), n_timesteps) #1D
obsdict.initialize_observable('U',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_bos',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_0',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_1',(1,), n_timesteps) #1D

def compute_n_system(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.conjugate(state) @ lat.sso('ch',1) @ lat.sso('c',1) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

def compute_n_3(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.conjugate(state) @ lat.sso('ch',3) @ lat.sso('c',3) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

def compute_U(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = om_0 * np.real( np.conjugate(state) @ lat.sso('ah',2) @ lat.sso('a',2) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

def compute_n_bos(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.conjugate(state) @ lat.sso('ah',2) @ lat.sso('a',2) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

def compute_n_0(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.conjugate(state) @ lat.sso('ch',0) @ lat.sso('c',0) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

def compute_n_1(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.conjugate(state) @ lat.sso('ch',1) @ lat.sso('c',1) @ state  )  
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('n_system',compute_n_system)
obsdict.add_observable_computing_function('n_3',compute_n_3)
obsdict.add_observable_computing_function('U',compute_U)
obsdict.add_observable_computing_function('n_bos',compute_n_bos)
obsdict.add_observable_computing_function('n_0',compute_n_0)
obsdict.add_observable_computing_function('n_1',compute_n_1)

#NOTE: von Neuman entropy is not computed here. It should be possible to compute it with syten

#compute QJ time evolution
os.chdir('data_qj_ed')
#init_state = lat.sso('ch',1) @ init_state #FIXME: remove this!!!!!!! 

ed_quantum_jumps = ed_quantum_jumps.EdQuantumJumps(4, h_tot , []  ) #l_list, [ lat.sso('ch',0), lat.sso('c',0) ]

first_trajectory = first_trajectory  #+ rank  NOTE: uncomment "+ rank" when parallelizing
#compute qj trajectories sequentially
for trajectory in range(first_trajectory, first_trajectory + n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = ed_quantum_jumps.quantum_jump_single_trajectory_time_evolution(init_state, t_max, dt, trajectory, obsdict )

#averages and errors
read_directory = os.getcwd()
write_directory = os.getcwd()

obsdict.compute_trajectories_averages_and_errors( list( range( first_trajectory, first_trajectory + n_trajectories) ), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 

#PLOT                                                                              
# n_system_av = np.loadtxt('n_system_av')

# plt.plot(time_v, n_system_av[:-1], label='n_system_av')

# plt.legend()
#plt.show()
