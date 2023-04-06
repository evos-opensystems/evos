"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with mps quantum jumps. The dimension of the oscillator needs to be strongly truncated.
"""
import evos.src.methods.mps_quantum_jumps as mps_quantum_jumps
import evos.src.observables.observables_pickled as observables
import pyten as ptn
import numpy as np 
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
#import math
import os
sys.stdout.write('test')

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

Gamma = 0.5
mu_l = +1
mu_r = +1
T_l = 1
T_r = 1
k_b = 1 #boltzmann constant
 
dt = 0.05
t_max = 10
time_v = np.arange(0, t_max, dt)
n_timesteps = int(t_max/dt)
n_trajectories = 1
first_trajectory = 2

#Lattice
ferm_bos_sites = [ 1, 1, 1, 1, 0, 1, 1 ] #doubled the fermionic sites to project-purify
lat = ptn.mp.lat.u1u1.genSpinlessFermiBose(ferm_bos_sites, max_bosons)
lat = ptn.mp.proj_pur.proj_purification(lat, [0], ["a", "ah"])
# print(lat)
vac_state =  ptn.mp.proj_pur.generateNearVacuumState(lat, 2, "0," + str( max_bosons ) )

#prepare PP vacuum for fermions
vac_state *= lat.get('ch',1)    
vac_state *= lat.get('ch',3) 
vac_state *= lat.get('ch',7)    
   
class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps):
        h_s = eps * lat.get('c',2) * lat.get('ch',2) #no need to PP
        #h_s = ptn.mp.addLog(h_s)
        h_s.truncate()
        return h_s 

    def h_b(self, Om_kl, Om_kr, mu_l, mu_r):
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = []
        h_b.append( ( Om_kl - mu_l ) * lat.get('c',0) * lat.get('ch',0) ) #no need to PP
        h_b.append( ( Om_kr - mu_r ) * lat.get('c',6) * lat.get('ch',6) ) #no need to PP
        h_b = ptn.mp.addLog(h_b)
        h_b.truncate()
        return h_b
   
    def h_t(self, g_kl, g_kr):
        #h_t = []
        h_t = g_kl * ( lat.get('ch',1) * lat.get('c',0) * lat.get('c',3) * lat.get('ch',2) + lat.get('c',2) * lat.get('ch',3) * lat.get('ch',0) * lat.get('c',1) ) 
        h_t += g_kr * ( lat.get('ch',7) * lat.get('c',6) * lat.get('c',3) * lat.get('ch',2) + lat.get('c',2) * lat.get('ch',3) * lat.get('ch',6) * lat.get('c',7) ) 
        #h_t = ptn.mp.addLog(h_t)
        h_t.truncate()
        return h_t
    
    def h_v(self, om_0, F):
        #FIXME: need to detract N0
        #FIXME: is m = 1 ?
        h_v = om_0 * lat.get('a',4) * lat.get('ah',4) - F * lat.get('c',2) * lat.get('ch',2) * ( ( lat.get('a',5) * lat.get('ah',4) + lat.get('ah',5) * lat.get('a',4) ) )
        h_v.truncate()
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, mu_l, mu_r, g_kl, g_kr, om_0, F):
        h_tot = self.h_s(eps) + self.h_b(Om_kl, Om_kr, mu_l, mu_r) + self.h_t(g_kl, g_kr) + self.h_v(om_0, F)
        h_tot.truncate()
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
    l_list_left.append( np.sqrt( Gamma * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l ) ) * lat.get( 'ch',1 ) * lat.get( 'c',0 ) )
    l_list_left.append( np.sqrt( Gamma * fermi_dist( 1./T_l, Om_kl, mu_l)) * lat.get('c',1) * lat.get('ch',0) )
    return l_list_left

def lindblad_op_list_right_lead( Om_kr, Gamma, mu_r, T_r ):
    l_list_right = []
    l_list_right.append( np.sqrt( Gamma * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) ) * lat.get( 'ch',7 ) * lat.get( 'c',6 ) )
    l_list_right.append( np.sqrt( Gamma * fermi_dist( 1./T_r, Om_kr, mu_r)) * lat.get('c',7) * lat.get('ch',6) )
    return l_list_right

l_list_left = lindblad_op_list_left_lead( Om_kl, Gamma, mu_l, T_l )
l_list_right = lindblad_op_list_right_lead( Om_kr, Gamma, mu_r, T_r )
l_list = l_list_left + l_list_right

#Observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n',(8,), n_timesteps) 
obsdict.initialize_observable('block_entropies',(7,), n_timesteps)
obsdict.initialize_observable('rdm_phon',(max_bosons + 1, max_bosons + 1), n_timesteps)

def compute_n(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    for site in range(8):
        obs_array[site] = np.real( ptn.mp.expectation(state, lat.get('n', site) ) ) #/ state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

def compute_block_entropies(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = state.block_entropies()
    #OBS DEPENDENT PART END
    return obs_array

def compute_rdm_phon(state, obs_array_shape,dtype = 'complex'):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = rdm =  np.array(ptn.mp.rdm.o1rdm(state,4) )
    #OBS DEPENDENT PART END
    return obs_array


obsdict.add_observable_computing_function('n', compute_n )
obsdict.add_observable_computing_function('block_entropies', compute_block_entropies )
obsdict.add_observable_computing_function('rdm_phon', compute_rdm_phon )

########TDVP CONFIG
conf_tdvp = ptn.tdvp.Conf()
conf_tdvp.mode = ptn.tdvp.Mode.GSE 
conf_tdvp.dt = dt
conf_tdvp.trunc.threshold = 1e-8  #NOTE: set to zero for gse
conf_tdvp.trunc.weight = 1e-10 #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp.trunc.maxStates = 1000
conf_tdvp.exp_conf.errTolerance = 1e-7
conf_tdvp.exp_conf.inxTolerance = 1e-6
conf_tdvp.exp_conf.maxIter =  10
conf_tdvp.cache = 1
conf_tdvp.maxt = t_max
conf_tdvp.gse_conf.krylov_order = 3 
conf_tdvp.gse_conf.trunc_op = ptn.Truncation(1e-8 , maxStates=500) #maxStates shuld be the same as the one used for tdvp! 1e-8 - 1e-6
conf_tdvp.gse_conf.trunc_expansion = ptn.Truncation(1e-6, maxStates=500) #precision of GSE. par is trunc. treshold. do not goe below 10^-12 (numerical instability)!!
conf_tdvp.gse_conf.adaptive = True
conf_tdvp.gse_conf.sing_val_thresholds = [1e-12] #most highly model-dependet parameter 


#compute time-evolution for one trajectory
#vac_state *= lat.get('c',3) * lat.get('ch',2) #FIXME: remove this!!!!!!!

#exite one particle in the left lead and one in the right lead
vac_state *= lat.get('c',1) * lat.get('ch',0)
vac_state *= lat.get('c',7) * lat.get('ch',6)

qj = mps_quantum_jumps.MPSQuantumJumps(8, lat, h_tot, []) #l_list

os.chdir('data_qj_mps')
first_trajectory = first_trajectory  #+ rank  NOTE: uncomment "+ rank" when parallelizing

#COMPUTE ONE TRAJECTORY WITH TDVP 
for trajectory in range(first_trajectory, first_trajectory + n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(vac_state, conf_tdvp, t_max, dt, trajectory, obsdict)

read_directory = os.getcwd()
write_directory = os.getcwd()


obsdict.compute_trajectories_averages_and_errors( list( range( first_trajectory, first_trajectory + n_trajectories) ), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 

#PLOT
n_av = np.load('n_av.npy')

fig, ax = plt.subplots()
ax.plot(time_v, n_av[2,:-1], label='n_sys')

plt.legend()
fig.savefig('test_mps.png')
#########################