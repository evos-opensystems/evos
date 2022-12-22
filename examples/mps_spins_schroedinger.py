"""_summary_

    Returns
    -------
    _type_
        _description_
    """
import numpy as np
import time
import os
import psutil
import sys

import evos
import evos.src.methods.mps_schroedinger as mps_schroedinger
import evos.src.observables.observables as observables
import pyten as ptn

#FOR PARALLELIZING
# from mpi4py import MPI
# comm =MPI.COMM_WORLD
# rank = comm.rank
# size = comm.size

##PARAMETERS
n_sites = 4
J = 2
W = 10
seed_W = 7
n_timesteps_tot = 160 #FIXME: This is needed to initialize observable arrays!
dim_H = 2 ** n_sites  
rng = np.random.default_rng(seed=seed_W) # random numbers
eps_vec = rng.uniform(0, W, n_sites) #onsite disordered energy random numbers
spin = 0.5    
## 
#lattice 
lat = ptn.mp.lat.nil.genSpinLattice(n_sites, spin)

#Hamiltonian without fsm
h = []
#spin coupling
for i in range(n_sites):
    for j in range(i):
        h.append( J/np.abs(i-j)**3 * ( lat.get('sp', i) * lat.get('sm', j) +  lat.get('sp', j) * lat.get('sm', i) ) )
#disorder
for i in range(n_sites):
    h.append( eps_vec[i] * lat.get('sz', i) )

H = ptn.mp.addLog(h)

#generate antiferromagnetic state
init_state =  ptn.mp.generateNearVacuumState(lat)
for i in range(n_sites):
    init_state *= lat.get( "sm", i)
    init_state.truncate()
    
    if i % 2 == 0:
        init_state *= lat.get( "sx", i)
        init_state.normalise()
        init_state.truncate()
    print('exp value of sz on ',i, ptn.mp.expectation( init_state, lat.get('sz',i) ))

#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('sz',(n_sites,), n_timesteps_tot) #2D
obsdict.initialize_observable('bdim_mat',(n_sites,), n_timesteps_tot)  #2D

def compute_sz(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    for site in range(n_sites):
        obs_array[site] = np.real( ptn.mp.expectation(state, lat.get('sz', site) ) ) / state.norm() ** 2 #NOTE: state is in general not normalized
    
    #OBS DEPENDENT PART END
    return obs_array

def compute_bdim_mat(state, obs_array_shape,dtype):  
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART STAR
    for site in range(len(obs_array)):
        obs_array[site] = state[site].getTotalDims()[1] / state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('sz',compute_sz )
obsdict.add_observable_computing_function('bdim_mat',compute_bdim_mat )



#Time-evolution methods: 
# - global krylov from 0 to 1 with dt= 0.05
# - gse tdvp from 1 to 2 with dt= 0.06
# - 2 site tdvp from 2 to 4 with dt= 0.07
# - 1 site tdvp from 4 to 8 with dt= 0.08


#global krylov from 0 to 1 with dt= 0.05
conf_krylov = ptn.krylov.Conf()
conf_krylov.dt = 0.05
conf_krylov.threshold= 1e-8
conf_krylov.weight= 1e-10
conf_krylov.errTolerance= 1e-7
conf_krylov.inxTolerance= 1e-6
conf_krylov.maxIter = 100
conf_krylov.maxStates = 1000
conf_krylov.tend = 1

#tdvp_config 1: 2-site tdvp from 1 to 2 with dt= 0.06
conf_tdvp1 = ptn.tdvp.Conf()
conf_tdvp1.mode = ptn.tdvp.Mode.TwoSite
conf_tdvp1.dt = 0.05
conf_tdvp1.trunc.threshold = 1e-8  #NOTE: set to zero for gse
conf_tdvp1.trunc.weight = 1e-10 #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp1.trunc.maxStates = 1000
conf_tdvp1.exp_conf.errTolerance = 1e-7
conf_tdvp1.exp_conf.inxTolerance = 1e-6
conf_tdvp1.exp_conf.maxIter =  100
conf_tdvp1.cache = True
conf_tdvp1.maxt = 1 # 2-1


# tdvp_config 2: GSE tdvp from 2 to 4 with dt= 0.07
conf_tdvp2 = ptn.tdvp.Conf()
conf_tdvp2.mode = ptn.tdvp.Mode.GSE
conf_tdvp2.dt = 0.05
conf_tdvp2.trunc.threshold =0  #NOTE: set to zero for gse
conf_tdvp2.trunc.weight = 0 #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp2.trunc.maxStates = 1000
conf_tdvp2.exp_conf.errTolerance = 1e-7
conf_tdvp2.exp_conf.inxTolerance = 1e-6
conf_tdvp2.exp_conf.maxIter =  100
conf_tdvp2.cache = True
conf_tdvp2.maxt = 2 # 4-2
conf_tdvp2.gse_conf.krylov_order = 3
conf_tdvp2.gse_conf.trunc_op = ptn.Truncation(1e-8, maxStates=1000) #maxStates shuld be the same as the one used for tdvp! 1e-8 - 1e-6
conf_tdvp2.gse_conf.trunc_expansion = ptn.Truncation(1e-6, maxStates=1000) #precision of GSE. par is trunc. treshold. do not goe below 10^-12 (numerical instability)!!
conf_tdvp2.gse_conf.adaptive = True
conf_tdvp2.gse_conf.sing_val_threshold = 1e-12 #most highly model-dependet parameter

#tdvp_config 3: 1 site tdvp from 4 to 8 with dt= 0.08
conf_tdvp3 = ptn.tdvp.Conf()
conf_tdvp3.mode = ptn.tdvp.Mode.Single
conf_tdvp3.dt = 0.05
conf_tdvp3.trunc.threshold = 1e-8  #NOTE: set to zero for gse
conf_tdvp3.trunc.weight = 1e-10 #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp3.trunc.maxStates = 1000
conf_tdvp3.exp_conf.errTolerance = 1e-7
conf_tdvp3.exp_conf.inxTolerance = 1e-6
conf_tdvp3.exp_conf.maxIter =  100
conf_tdvp3.cache = True
conf_tdvp3.maxt = 4 # 8-4

#run time evolution
tdvp_config_list = [conf_tdvp1, conf_tdvp2, conf_tdvp3]
mps_schroedinger.MPSSchroedinger( n_sites, lat, H ).schroedinger_time_evolution( init_state, obsdict, tdvp_config_list, krylov = True, krylov_config = conf_krylov )

#plot and compare with ed_spins_schroedinger.py

