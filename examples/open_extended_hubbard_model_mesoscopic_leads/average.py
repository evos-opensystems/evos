import numpy as np
import time
import os
import psutil
import sys
import math
import evos.src.observables.observables as observables
import sys
#import evos.src.methods.mps_quantum_jumps_no_normalization_adaptive_timestep as mps_quantum_jumps_no_normalization_adaptive_timestep

#FOR PARALLELIZING
# from mpi4py import MPI
# comm =MPI.COMM_WORLD
# rank = comm.rank
# size = comm.size

name = 'ed_qj_meso_cluster'
#os.mkdir(name)
os.chdir(name)

n_sites = 2
n_leads_left = 1
n_leads_right = 1


n_tot = n_sites + n_leads_left + n_leads_right

tdvp_maxt = 1
tdvp_dt = 0.001

n_timesteps = int(tdvp_maxt/tdvp_dt)


#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n',(n_tot,), n_timesteps) #2D





obsdict.compute_trajectories_averages_and_errors( list(range(1000)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 

