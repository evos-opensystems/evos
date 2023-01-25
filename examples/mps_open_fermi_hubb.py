"""Example of quantum jumps with matrix-product states (MPS) using an adaptive-timestep time-dependent variational principle (TDVP)
The model is a spin chain and a single trajectory is computed.
"""
import numpy as np
import time
import os
import pyten as ptn
import psutil
import sys

import evos
import evos.src.methods.mps_quantum_jumps_no_normalization_adaptive_timestep as mps_quantum_jumps_no_normalization_adaptive_timestep
import evos.src.observables.observables as observables
import pyten as ptn

#FOR PARALLELIZING
# from mpi4py import MPI
# comm =MPI.COMM_WORLD
# rank = comm.rank
# size = comm.size

##PARAMETERS
n_sites_spinful = 3
n_sites = 2 * n_sites_spinful

U = 1
V = 1

J = 2
gamma = 1
W = 10
seed_W = 7
first_trajectory = 0
n_trajectories_average = 200
n_trajectories = 2
tdvp_maxt = 10
tdvp_dt = 0.01
tdvp_mode = 2
tdvp_trunc_threshold = 1e-8
tdvp_trunc_weight = 1e-10
tdvp_trunc_maxStates = 500
tdvp_exp_conf_errTolerance = 1e-7 
tdvp_exp_conf_inxTolerance = 1e-6
tdvp_exp_conf_maxIter = 10
tdvp_cache = 1 

tdvp_gse_conf_krylov_order = 3
tdvp_gse_conf_trunc_op_treshold = 1e-8
tdvp_gse_conf_trunc_expansion_treshold = 1e-6 
tdvp_gse_conf_trunc_expansion_maxStates = 1500
tdvp_gse_conf_adaptive = True
tdvp_gse_conf_sing_val_threshold = 1e-12

dim_H = 2 ** n_sites  
rng = np.random.default_rng(seed=seed_W) # random numbers
eps_vec = rng.uniform(0, W, n_sites) #onsite disordered energy random numbers
spin = 0.5    
tol = 1e-4 #for bisection method in adaptive timestep tdvp
max_iterations = 10 #for bisection method in adaptive timestep tdvp
n_timesteps = int(tdvp_maxt/tdvp_dt)
## 
#lattice 
lat = ptn.mp.lat.nil.genFermiLattice(n_sites)
#help(ptn.mp.lat.nil.genFermiLattice)
#print(lat)

h = []
#HOPPING
for i in range(0, n_sites-2, 2):
    #print(i, i+2)
    #print(i+1, i+3)
    h.append(-U*(  lat.get('c', i+2) * lat.get('ch', i) +  lat.get('c', i) * lat.get('ch', i+2) ) )  # even terms = spin up
    h.append(-U*(  lat.get('c', i+1) * lat.get('ch', i+3) +  lat.get('c', i+3) * lat.get('ch', i+1) )  ) # odd terms = spin down

#COULOMB
for i in range(0, n_sites, 2):
    #print(i,i+1)
    h.append(V* (  lat.get('n', i) * lat.get('n', i+1) ) )

H = ptn.mp.addLog(h)
#print(h)

# Lindbladian

L = []
for i in range(0, n_sites, 2): 
    print(i, i+1)
    L.append(lat.get('c', i) + lat.get('n', i+1) )

init_state =  ptn.mp.generateNearVacuumState(lat)
for i in range(0,n_sites, 4):
    print(i)
    init_state *= lat.get( "ch", i)
    init_state.normalise()
    init_state.truncate()

    print('exp value of n on ',i, ptn.mp.expectation( init_state, lat.get('n',i) ))

for i in range(3,n_sites, 4):
    print(i)
    init_state *= lat.get( "ch", i)
    init_state.normalise()
    init_state.truncate()
    print('exp value of n on ',i, ptn.mp.expectation( init_state, lat.get('n',i) ))
     

qj = mps_quantum_jumps_no_normalization_adaptive_timestep.MPSQuantumJumps(n_sites, lat, H, L) #ADAPTIVE TIMESTEP, NO NORMALIZATION

#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n',(n_sites,), n_timesteps) #2D
obsdict.initialize_observable('bdim_mat',(n_sites,), n_timesteps)  #2D

def compute_n(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    for site in range(n_sites):
        obs_array[site] = np.real( ptn.mp.expectation(state, lat.get('n', site) ) ) / state.norm() ** 2 #NOTE: state is in general not normalized
    
    #OBS DEPENDENT PART END
    return obs_array

def compute_bdim_mat(state, obs_array_shape,dtype):  
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART STAR
    for site in range(len(obs_array)):
        obs_array[site] = state[site].getTotalDims()[1] / state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('n',compute_n )
obsdict.add_observable_computing_function('bdim_mat',compute_bdim_mat )

########TDVP CONFIG
conf_tdvp = ptn.tdvp.Conf()
conf_tdvp.mode = ptn.tdvp.Mode.TwoSite 
conf_tdvp.dt = tdvp_dt
conf_tdvp.trunc.threshold = tdvp_trunc_threshold  #NOTE: set to zero for gse
conf_tdvp.trunc.weight = tdvp_trunc_weight #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp.trunc.maxStates = tdvp_trunc_maxStates
conf_tdvp.exp_conf.errTolerance = tdvp_exp_conf_errTolerance
conf_tdvp.exp_conf.inxTolerance = tdvp_exp_conf_inxTolerance
conf_tdvp.exp_conf.maxIter =  tdvp_exp_conf_maxIter
conf_tdvp.cache = tdvp_cache
conf_tdvp.maxt = tdvp_maxt


#COMPUTE ONE TRAJECTORY WITH TDVP AND ADAPTIVE TIMESTEP
#test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(init_state, conf_tdvp, tdvp_maxt, tdvp_dt, tol, max_iterations, trajectory, obsdict, tdvp_trunc_threshold, tdvp_trunc_weight, tdvp_trunc_maxStates)


for trajectory in range(n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(init_state, conf_tdvp, tdvp_maxt, tdvp_dt, tol, max_iterations, trajectory, obsdict, tdvp_trunc_threshold, tdvp_trunc_weight, tdvp_trunc_maxStates)


read_directory = os.getcwd()
write_directory = os.getcwd()


obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 



