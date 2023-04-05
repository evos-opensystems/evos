"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with mps quantum jumps. The dimension of the oscillator needs to be strongly truncated.
"""
import evos.src.methods.mps_quantum_jumps as mps_quantum_jumps
import evos.src.observables.observables as observables
import pyten as ptn
import numpy as np 
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
#import math
import os

#PARAMETERS
max_bosons = 2
# eps = 1
# Om_kl = 1
# Om_kr = 1
# g_kl = 1
# g_kr = 1
# om_0 = 1
# F = 1
# #FIXME: how do I compute N0??

# Gamma = 1
# mu_l = +1
# mu_r = +1
# T_l = 1
# T_r = 1
# k_b = 1 #boltzmann constant
 
dt = 0.05
t_max = 8
time_v = np.arange(0, t_max, dt)
n_timesteps = int(t_max/dt)
n_trajectories = 50
trajectory = 0
first_trajectory = 0

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
   
#print(vac_state.norm())
# for site in range(8):
#     print('n on site {} is {}'.format( site, ptn.mp.expectation(vac_state, lat.get('n',site ) ) ) )

##########################PREPARE A THERMAL STATE ON SITE 1 TO TEST FERMIONIC-PP
#parameters
beta = 1
T = 1
omega = 1

#hamiltonian
h = lat.get('I') #[lat.get('nf',0)] #hermitian, thus no balancing operators needed
#lindblad operators
l_list = [ np.exp( - beta * omega/2) * lat.get('c', 1) * lat.get('ch', 0), lat.get('ch',1) * lat.get('c',0) ]

#Observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('nf_0',(1,), n_timesteps) #1D
obsdict.initialize_observable('nf_1',(1,), n_timesteps) #1D

def compute_nf_0(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = np.real( ptn.mp.expectation(state, lat.get('nf', 0) ) ) / state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

def compute_nf_1(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = np.real( ptn.mp.expectation(state, lat.get('nf', 1) ) ) / state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('nf_0',compute_nf_0 )
obsdict.add_observable_computing_function('nf_1',compute_nf_1 )

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
qj = mps_quantum_jumps.MPSQuantumJumps(8, lat, h, l_list) #ADAPTIVE TIMESTEP, NO NORMALIZATION

os.chdir('data_qj_mps')
trajectory = first_trajectory  #+ rank  NOTE: uncomment "+ rank" when parallelizing
print('computing time-evolution for trajectory {}'.format(trajectory) )

#COMPUTE ONE TRAJECTORY WITH TDVP 
for trajectory in range(n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(vac_state, conf_tdvp, t_max, dt, trajectory, obsdict)

read_directory = os.getcwd()
write_directory = os.getcwd()


obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 
#exact population of thermal state

n_f_exact = 1./( 1 + np.exp( + beta * omega) )

#PLOT
nf_0_av = np.loadtxt('nf_0_av')
nf_1_av = np.loadtxt('nf_1_av')

fig, ax = plt.subplots()
ax.plot(time_v, nf_0_av[:-1], label='nf_0_av')
ax.plot(time_v, nf_1_av[:-1], label='nf_1_av')
ax.plot(time_v, nf_0_av[:-1] + nf_1_av[:-1], label='nf_0_av + nf_1_av')

ax.hlines(y=n_f_exact, xmin=0, xmax=t_max, linewidth=2, color='r')
plt.legend()
fig.savefig('test_mps.png')
#########################