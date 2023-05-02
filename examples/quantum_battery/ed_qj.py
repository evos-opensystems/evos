"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with ed quantum jumps. The dimension of the oscillator needs to be strongly truncated.
"""

import evos.src.lattice.dot_with_oscillator_lattice as lattice 
import evos.src.methods.partial_traces.partial_trace_tls_boson as pt 
import evos.src.methods.ed_quantum_jumps as ed_quantum_jumps
import evos.src.observables.observables as observables
import evos.src.methods.partial_traces.partial_trace_tls_boson as pt 

import numpy as np 
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as la
from scipy import linalg as sla
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

om_0 = 1 #0.2
m = 1
lamb = 1 #0.1
x0 = np.sqrt( 2./ (m * om_0) )
F = 2 *lamb / x0

eps = 0  
Om_kl = +0.5
Om_kr = -0.5
Gamma = 2
g_kl = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
g_kr = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
N0 = 0.5 #FIXME: is this correct?
delta_l = 1.
delta_r = 1.

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
first_trajectory = 4

################
def make_writing_dir_and_change_to_it( parent_data_dirname: str, parameter_dict: dict, overwrite: bool = False, create_directory: bool = True ) -> str :
    """given a dictionary with some selected job's parameters, it creates the correct subfolder in which to run the job and changes to it

    Parameters
    ----------
    parent_data_dirname : str
        name of the parent directory
    parameter_dict : dict
        parameter dictionary specifying the directory

    Returns
    -------
    str
        path of the directory in which to write the states or the observables
    """
    import os 
    from datetime import date

    #go to parent folder if existing. create one with date attached and go to it if not existing
    if os.path.isdir(parent_data_dirname):
            os.chdir(parent_data_dirname)
    else:
        if create_directory:
            parent_data_dirname += '_'+str( date.today() )
            os.mkdir( parent_data_dirname )
            os.chdir(parent_data_dirname)


    dir_depth = len(parameter_dict)
    count_dir_depth = 0
    for par in parameter_dict:
        subdir_name = par +'_'+str(parameter_dict[par])

        #if reached lowest directory level AND it already exists
        if count_dir_depth == dir_depth-1 and os.path.isdir(subdir_name): 
            #print(subdir_name)
            if not overwrite and create_directory: 
                subdir_name += '_'+str( date.today() )
        
        #all other directory levels OR the lowest but it doesn't exists
        if os.path.isdir(subdir_name):
            os.chdir(subdir_name)
        else:
            if create_directory:
                os.mkdir( subdir_name )
                os.chdir(subdir_name)
                
        count_dir_depth += 1

    writing_dir = os.getcwd()
    return writing_dir

parameter_dict = {'max_bosons': max_bosons, 'dt': dt, 't_max': t_max, 'mu_l' : mu_l, 'mu_r' : mu_r, 'n_trajectories' : n_trajectories, 'first_trajectory' : first_trajectory   }
writing_dir = make_writing_dir_and_change_to_it('data_qj_ed', parameter_dict, overwrite=True)

################


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
        h_b = Om_kl * lat.sso('ch',0) @ lat.sso('c',0) + Om_kr * lat.sso('ch',3) @ lat.sso('c',3)
        return h_b
   
    def h_t(self, g_kl, g_kr): #system-leads
        h_t = g_kl * ( lat.sso('ch',1) @ lat.sso('c',0) + lat.sso('ch',0) @ lat.sso('c',1) ) + g_kr * ( lat.sso('ch',1) @ lat.sso('c',3) + lat.sso('ch',3) @ lat.sso('c',1) )
        return h_t
    
    def h_boson(self, om_0): #oscillator
        h_boson = om_0 * lat.sso('ah',2) @ lat.sso('a',2)
        return h_boson
    
    def h_v(self, F, N0): #system-oscillator
        dimH = lat.sso('c',0).shape[0]
        h_v = - F * ( lat.sso('ch',1) @ lat.sso('c',1) - N0 * np.eye(dimH, dtype='complex') ) @ ( lat.sso('ah',2) + lat.sso('a',2) ) 
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F):
        h_tot = self.h_boson(om_0) + self.h_v(F, N0) +self.h_s(eps) + self.h_t(g_kl, g_kr) + self.h_b(Om_kl, Om_kr)
        return h_tot
        
 
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)
h_tot = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F)
# print('h_tot is symmetric: ', ( h_tot == h_tot.T ).all() )
# quit()

#Lindblad operators
def fermi_dist(beta, e, mu):
    f = 1. / ( np.exp( beta * (e-mu) ) + 1.)
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

###########    FIXME: FOR DEBUGGING !!!!!!!!!
# occ_state = init_state.copy()
# occ_state = lat.sso('ch',0) @ occ_state
# occ_state /= la.norm(occ_state)
# occ_state = lat.sso('ch',1) @ occ_state
# occ_state /= la.norm(occ_state)
# occ_state = lat.sso('ah',2) @ occ_state
# occ_state /= la.norm(occ_state)
# occ_state = lat.sso('ch',3) @ occ_state
# occ_state /= la.norm(occ_state)

# init_state = init_state + occ_state
# init_state /= la.norm(init_state)
###########    

#Observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n_system',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_3',(1,), n_timesteps) #1D
obsdict.initialize_observable('U',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_bos',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_0',(1,), n_timesteps) #1D
obsdict.initialize_observable('n_1',(1,), n_timesteps) #1D
obsdict.initialize_observable('free_energy_neq',(1,), n_timesteps)
obsdict.initialize_observable('sec_ord_coherence_funct',(1,), n_timesteps) 

def compute_n_system(state, obs_array_shape,dtype): 
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.conjugate(state) @ lat.sso('ch',1) @ lat.sso('c',1) @ state)  
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

def compute_free_energy_neq(state, obs_array_shape,dtype):  
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART STAR
    #phonon energy
    phonon_energy =  om_0 * np.real( np.conjugate(state) @ lat.sso('ah',2) @ lat.sso('a',2) @ state  )
    #phononic RDM
    rho_tot = np.outer( np.conjugate(state), state)
    rho123 = pt.tracing_out_one_tls_from_tls_bosonic_system(0, rho_tot, [1,1,0,1], max_bosons)
    rho23 = pt.tracing_out_one_tls_from_tls_bosonic_system(0, rho123, [1,0,1], max_bosons)
    rdm = pt.tracing_out_one_tls_from_tls_bosonic_system(1, rho23, [0,1], max_bosons)
    #von Neumann entropy
    R = rdm * ( sla.logm( rdm )/ sla.logm( np.matrix( [ [ 2 ] ] ) ) )
    S = - np.matrix.trace(R)
    #non-eq. free energy
    obs_array = phonon_energy - T_l * S
    #OBS DEPENDENT PART END
    return obs_array

def compute_sec_ord_coherence_funct(state, obs_array_shape,dtype):  
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART STAR
    #phononic RDM
    rho_tot = np.outer( np.conjugate(state), state)
    rho123 = pt.tracing_out_one_tls_from_tls_bosonic_system(0, rho_tot, [1,1,0,1], max_bosons)
    rho23 = pt.tracing_out_one_tls_from_tls_bosonic_system(0, rho123, [1,0,1], max_bosons)
    rdm = pt.tracing_out_one_tls_from_tls_bosonic_system(1, rho23, [0,1], max_bosons)
    #g_2
    numerator, denominator = 0., 0.
    for mode in range(rdm.shape[0]): #'max_bosons+1'
        numerator += mode * (mode - 1) * rdm[ mode, mode ]
        denominator += (mode * rdm[ mode, mode ])
    denominator = denominator ** 2
    obs_array = numerator/denominator
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('n_system',compute_n_system)
obsdict.add_observable_computing_function('n_3',compute_n_3)
obsdict.add_observable_computing_function('U',compute_U)
obsdict.add_observable_computing_function('n_bos',compute_n_bos)
obsdict.add_observable_computing_function('n_0',compute_n_0)
obsdict.add_observable_computing_function('n_1',compute_n_1)
obsdict.add_observable_computing_function('free_energy_neq', compute_free_energy_neq )
obsdict.add_observable_computing_function('sec_ord_coherence_funct', compute_sec_ord_coherence_funct )

#NOTE: von Neuman entropy is not computed here. It should be possible to compute it with syten

#compute QJ time evolution

ed_quantum_jumps = ed_quantum_jumps.EdQuantumJumps(4, h_tot , l_list  ) #l_list, [ lat.sso('ch',0), lat.sso('c',0) ]

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
