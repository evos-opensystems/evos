import pyten as ptn
import numpy as np
import time
import os
import psutil
import sys
import math
import evos.src.observables.observables as observables
import evos.src.methods.mps_quantum_jumps as mps_quantum_jumps
#import evos.src.methods.mps_quantum_jumps_no_normalization_adaptive_timestep as mps_quantum_jumps_no_normalization_adaptive_timestep

#FOR PARALLELIZING
# from mpi4py import MPI
# comm =MPI.COMM_WORLD
# rank = comm.rank
# size = comm.size

name = 'mps_test_subspace_tmax30_2l_2s_2l_L0.3_2'
os.mkdir(name)
os.chdir(name)

#############################################################################################
# PARAMETERS FOR SYSTEM + LEADS
n_sys_sites_spinful = 2
n_leads_left_spinful = 2
n_leads_right_spinful = 2

n_sites = 2 * n_sys_sites_spinful
n_leads_left = 2 * n_leads_left_spinful
n_leads_right = 2 * n_leads_right_spinful

n_tot = n_sites + n_leads_left + n_leads_right


dim_H_sys = 2 ** n_sites

dim_H_lead_left =  2 ** n_leads_left 
dim_H_lead_right =  2** n_leads_right

dim_tot = dim_H_sys*dim_H_lead_left*dim_H_lead_right

# coupling parameters
t_hop = 1
U = 1
V = 1

# spectral function
eps = 1
kappa = 1

# lead coefficients
alpha = 1

# temperature and chemical potential on the different leads
T_L = 1
T_R = 1
mu_L = 1
mu_R = -1


def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

####################################################################################################################
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
eps_step_l = 2* W / n_leads_left_spinful 
eps_vector_l = np.arange( -W, W, eps_step_l )
eps_delta_vector_l = eps_step_l * np.ones( len(eps_vector_l) )

k_vector_l = np.zeros( len(eps_vector_l) )
for i in range( len(eps_vector_l) ):
    k_vector_l[i] = np.sqrt( const_spec_funct( G , W, eps_vector_l[i] ) * eps_delta_vector_l[i]/ (2*math.pi) )  
    
#RIGHT LEAD COEEFICIENTS
eps_step_r = 2* W / n_leads_right_spinful
eps_vector_r = np.arange( -W, W, eps_step_r )
eps_delta_vector_r = eps_step_r * np.ones( len(eps_vector_r) )

k_vector_r = np.zeros( len(eps_vector_r) )
for i in range( len(eps_vector_r) ):
    k_vector_r[i] = np.sqrt( const_spec_funct( G , W, eps_vector_r[i] ) * eps_delta_vector_r[i]/ (2*math.pi) ) 
    


print('kvec =', k_vector_l)
print('kvec right =', k_vector_r)

#############################################################################################
# PARAMTERS FOR TDVP SOLVER
gamma = 0
W = 10
seed_W = 2
first_trajectory = 0
#n_trajectories_average = 200
n_trajectories = 1
tdvp_maxt = 30
tdvp_dt = 0.005
tdvp_mode = 2
tdvp_trunc_threshold = 1e-9
tdvp_trunc_weight = 1e-10
tdvp_trunc_maxStates = 100000
tdvp_exp_conf_errTolerance =  1e-8 #1e-7 
tdvp_exp_conf_inxTolerance = 1e-7 #1e-6
tdvp_exp_conf_maxIter = 10
tdvp_cache = 1 

tdvp_gse_conf_krylov_order = 5 # 3
tdvp_gse_conf_trunc_op_treshold = 1e-8
tdvp_gse_conf_trunc_expansion_treshold = 1e-6
tdvp_gse_conf_trunc_expansion_maxStates = 100000
tdvp_gse_conf_adaptive = True
tdvp_gse_conf_sing_val_threshold = 1e-12


rng = np.random.default_rng(seed=seed_W) # random numbers
#eps_vec = rng.uniform(0, W, n_tot) #onsite disordered energy random numbers
#spin = 0.5    
tol = 1e-4 #for bisection method in adaptive timestep tdvp
max_iterations = 10 #for bisection method in adaptive timestep tdvp
n_timesteps = int(tdvp_maxt/tdvp_dt)
## 
#lattice 
lat = ptn.mp.lat.nil.genFermiLattice(n_tot)
#help(ptn.mp.lat.nil.genFermiLattice)
#print(lat)

#############################################################################################################
# HAMILTONIAN
print(n_tot)

# SYSTEM
h = []
#HOPPING
for i in range(n_leads_left, n_leads_left + n_sites-2, 2):
    #print(i, i+2)
    #print(i+1, i+3)
    h.append(-t_hop*(   lat.get('ch', i) * lat.get('c', i+2)  +   lat.get('ch', i+2) * lat.get('c', i) ) )  # even terms = spin up
    h.append(-t_hop*(   lat.get('ch', i+3) * lat.get('c', i+1) +  lat.get('ch', i+1) * lat.get('c', i+3)  )  ) # odd terms = spin down

# COULOMB
for i in range(n_leads_left, n_leads_left + n_sites, 2):
    #print(i,i+1)
    h.append(U* (  lat.get('n', i) * lat.get('n', i+1) ) )

# COULOMB NEAREST NEIGHBOUR
for i in range(n_leads_left, n_leads_left + n_sites -2, 2):
    print(i,i+2)
    print(i,i+3)
    print(i+1,i+2)
    print(i+1,i+3)
    h.append(V* (  lat.get('n', i) * lat.get('n', i+2) ) )
    h.append(V* (  lat.get('n', i) * lat.get('n', i+3) ) )
    h.append(V* (  lat.get('n', i+1) * lat.get('n', i+2) ) )
    h.append(V* (  lat.get('n', i+1) * lat.get('n', i+3) ) )

# LEADS 

# LEFT
# E_kin left
m = 0
for i in range(0, n_leads_left,2):
    print('ekin left index', i,i+1)
    h.append( (eps_vector_l[m] ) * (lat.get('n', i) + lat.get('n', i+1)) )
    print('m left', m)
    m = m+1
    #print(i)

# hopping system lead left
m = 0
for i in range(0, n_leads_left,2):
    print(i,n_leads_left)
    print(i+1,n_leads_left+1)
    print(k_vector_l[m])
    h.append( k_vector_l[m] * (  lat.get('ch', i) * lat.get('c', n_leads_left)  +  lat.get('ch', n_leads_left) *  lat.get('c', i) ) )
    h.append( k_vector_l[m] * (  lat.get('ch', n_leads_left+1) * lat.get('c', i+1) +   lat.get('ch', i+1) * lat.get('c', n_leads_left+1) ) )
    m = m + 1 

   
# RIGHT
# E_kin right

m = 0
for i in range(n_leads_left+n_sites, n_tot,2):
    print('ekin right index', i,i+1)
    print('m right', m)
    h.append( (eps_vector_r[m] ) * (lat.get('n', i) + lat.get('n', i+1)) )
    #print(m)  
    m = m + 1   

# hopping system lead right
m = 0
for i in range(n_leads_left+n_sites, n_tot,2):
    print(i,n_leads_left+n_sites-2)
    print(i+1,n_leads_left+n_sites-1)
    print('kright', k_vector_r[m])
    h.append( k_vector_r[m] * ( lat.get('ch', i) * lat.get('c', n_leads_left+n_sites-2)  +   lat.get('ch', n_leads_left+n_sites-2) * lat.get('c', i) ) )
    h.append( k_vector_r[m] * (  lat.get('ch', n_leads_left+n_sites-1) * lat.get('c', i+1) +   lat.get('ch', i+1) * lat.get('c', n_leads_left+n_sites-1)) )
    m = m+1


H = ptn.mp.addLog(h)

lat.add('H','H',H)
lat.save('lat')


# Lindbladian
alpha = 0.3

L_list = []
'''
m = 0
for i in range(0, n_leads_left,2):
    print(i,i+1)
    print('m = ', m)
    print('eps_delta', eps_delta_vector_l[m])
    print('eps', eps_vector_l[m])
    print('ferm_dist', fermi_dist(1/T_L, eps_vector_l[m], mu_L))
    print('exponential left =', np.exp( 1/T_L * (eps_vector_l[m] - mu_L) ))
    L_list.append(   lat.get('c', i)    * np.sqrt( eps_delta_vector_l[m] * np.exp( 1/T_L * (eps_vector_l[m] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[m], mu_L)) )
    L_list.append(   lat.get('c', i+1)  * np.sqrt( eps_delta_vector_l[m] * np.exp( 1/T_L * (eps_vector_l[m] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[m], mu_L)) )
    L_list.append(   lat.get('ch', i)   * np.sqrt( eps_delta_vector_l[m] * fermi_dist(1/T_L, eps_vector_l[m], mu_L) ) )
    L_list.append(   lat.get('ch', i+1) * np.sqrt( eps_delta_vector_l[m] * fermi_dist(1/T_L, eps_vector_l[m], mu_L) ) )
   
    m = m+1

print('alpha 1,2 = ', np.sqrt( eps_delta_vector_l[0] * np.exp( 1/T_L * (eps_vector_l[0] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[0], mu_L)))
print('alpha 3,4 = ', np.sqrt( eps_delta_vector_l[0] * fermi_dist(1/T_L, eps_vector_l[0], mu_L) ))

  
m = 0
for i in range(n_leads_left + n_sites, n_tot,2):
    print(i, i+1)
    print('m = ', m)
    print('eps_delta r', eps_delta_vector_r[m])
    print('eps r', eps_vector_r[m])
    print('ferm_dist r', fermi_dist(1/T_R, eps_vector_r[m], mu_R))
    print('exponential=', np.exp( 1/T_R * (eps_vector_r[m] - mu_R) ))
    L_list.append(  lat.get('c',i)     * np.sqrt( eps_delta_vector_r[m] * np.exp( 1/T_R * (eps_vector_r[m] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[m], mu_R)))
    L_list.append(  lat.get('c', i+1)  * np.sqrt( eps_delta_vector_r[m] * np.exp( 1/T_R * (eps_vector_r[m] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[m], mu_R)))
    L_list.append(  lat.get('ch', i)   * np.sqrt( eps_delta_vector_r[m] * fermi_dist(1/T_R, eps_vector_r[m], mu_R)) )
    L_list.append(  lat.get('ch', i+1) * np.sqrt( eps_delta_vector_r[m] * fermi_dist(1/T_R, eps_vector_r[m], mu_R)) )

    m = m+1

print('alpha 5,6 = ', np.sqrt( eps_delta_vector_r[0] * np.exp( 1/T_R * (eps_vector_r[0] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[0], mu_R)))
print('alpha 7,8 =', np.sqrt( eps_delta_vector_r[0] * fermi_dist(1/T_R, eps_vector_r[0], mu_R)))

'''
m = 0
for i in range(0, n_leads_left,2):
    print(i,i+1)
    print('m = ', m)
    print('eps_delta', eps_delta_vector_l[m])
    print('eps', eps_vector_l[m])
    print('ferm_dist', fermi_dist(1/T_L, eps_vector_l[m], mu_L))
    print('exponential left =', np.exp( 1/T_L * (eps_vector_l[m] - mu_L) ))
    L_list.append(  alpha* lat.get('c', i)  ) # * np.sqrt( eps_delta_vector_l[m] * np.exp( 1/T_L * (eps_vector_l[m] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[m], mu_L)) )
    L_list.append(  alpha* lat.get('c', i+1) ) # * np.sqrt( eps_delta_vector_l[m] * np.exp( 1/T_L * (eps_vector_l[m] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[m], mu_L)) )
    L_list.append(  alpha* lat.get('ch', i) )  #* np.sqrt( eps_delta_vector_l[m] * fermi_dist(1/T_L, eps_vector_l[m], mu_L) ) )
    L_list.append(  alpha* lat.get('ch', i+1) )#* np.sqrt( eps_delta_vector_l[m] * fermi_dist(1/T_L, eps_vector_l[m], mu_L) ) )
    m = m+1

  
m = 0
for i in range(n_leads_left + n_sites, n_tot,2):
    print(i, i+1)
    print('m = ', m)
    print('eps_delta r', eps_delta_vector_r[m])
    print('eps r', eps_vector_r[m])
    print('ferm_dist r', fermi_dist(1/T_R, eps_vector_r[m], mu_R))
    print('exponential=', np.exp( 1/T_R * (eps_vector_r[m] - mu_R) ))
    L_list.append( alpha* lat.get('c',i)  ) #   * np.sqrt( eps_delta_vector_r[m] * np.exp( 1/T_R * (eps_vector_r[m] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[m], mu_R)))
    L_list.append( alpha* lat.get('c', i+1) ) #  * np.sqrt( eps_delta_vector_r[m] * np.exp( 1/T_R * (eps_vector_r[m] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[m], mu_R)))
    L_list.append( alpha* lat.get('ch', i) ) #  * np.sqrt( eps_delta_vector_r[m] * fermi_dist(1/T_R, eps_vector_r[m], mu_R)) )
    L_list.append( alpha* lat.get('ch', i+1) ) #* np.sqrt( eps_delta_vector_r[m] * fermi_dist(1/T_R, eps_vector_r[m], mu_R)) )
    m = m+1

#lat.add('L','L',L_list[0])

init_state =  ptn.mp.generateNearVacuumState(lat)

for i in range(0, n_tot):
    print(i)
    init_state *= lat.get( "c", i)
    init_state.normalise()
'''
for i in range(0,n_tot):
    print(i)
    init_state *= lat.get( "ch", i)
    #init_state.normalise()
    #init_state.truncate()

    #print('exp value of n on ',i, ptn.mp.expectation( init_state, lat.get('n',i) ))
    #print('exp value of n on ',i+1, ptn.mp.expectation( init_state, lat.get('n',i+1) ))
'''
init_state *=  lat.get( "ch", 0) #+ lat.get("ch", 1))
init_state *=  lat.get( "ch", 1)
init_state *= lat.get( "ch", 6) #+ lat.get("ch", 7))
init_state *= lat.get( "ch", 7)
init_state.normalise()
for i in range(0,n_tot):
    print('exp value of n on ',i, ptn.mp.expectation( init_state, lat.get('n',i) ))



print('length Llist', len(L_list))
print('exp value of L on ', ptn.mp.expectation( init_state, L_list[0] ))
print('exp value of H on ', ptn.mp.expectation( init_state, H ))
#print('exp value of L on ', ptn.mp.expectation( init_state, L[0] ))
#print(L_list)
L = []


#qj = mps_quantum_jumps_no_normalization_adaptive_timestep.MPSQuantumJumps(n_tot, lat, H, L_list) #ADAPTIVE TIMESTEP, NO NORMALIZATION
qj = mps_quantum_jumps.MPSQuantumJumps(n_tot, lat, H, L_list)


#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n',(n_tot,), n_timesteps) #2D
obsdict.initialize_observable('H',(n_tot,), n_timesteps) #2D
obsdict.initialize_observable('bdim_mat',(n_tot,), n_timesteps)  #2D
obsdict.initialize_observable('norm',(1,), n_timesteps)  #2D

def compute_n(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    for site in range(n_tot):
        obs_array[site] = np.real( ptn.mp.expectation(state, lat.get('n', site) ) ) / state.norm() ** 2 #NOTE: state is in general not normalized
    
    #OBS DEPENDENT PART END
    return obs_array

def compute_H(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    for site in range(n_tot):
        obs_array[site] = np.real( ptn.mp.expectation(state, H ) ) / state.norm() ** 2 #NOTE: state is in general not normalized
    
    #OBS DEPENDENT PART END
    return obs_array

def compute_norm(state, obs_array_shape,dtype):  
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = state.norm() #NOTE: state is in general not normalized
    
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
obsdict.add_observable_computing_function('H',compute_H )
obsdict.add_observable_computing_function('norm',compute_norm )
obsdict.add_observable_computing_function('bdim_mat',compute_bdim_mat )


########TDVP CONFIG
conf_tdvp = ptn.tdvp.Conf()

#conf_tdvp.mode = ptn.tdvp.Mode.GSE #= ptn.tdvp.Mode.TwoSite 

conf_tdvp.mode = ptn.tdvp.Mode.Subspace
conf_tdvp.expandMaxBlocksize = 900
conf_tdvp.expansion_trunc = ptn.Truncation(1e-9, maxStates=700)

conf_tdvp.dt = tdvp_dt
conf_tdvp.trunc.threshold = tdvp_trunc_threshold  #NOTE: set to zero for gse
conf_tdvp.trunc.weight = tdvp_trunc_weight #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp.trunc.maxStates = tdvp_trunc_maxStates
conf_tdvp.exp_conf.errTolerance = tdvp_exp_conf_errTolerance
conf_tdvp.exp_conf.inxTolerance = tdvp_exp_conf_inxTolerance
conf_tdvp.exp_conf.maxIter =  tdvp_exp_conf_maxIter
conf_tdvp.cache = tdvp_cache
conf_tdvp.maxt = tdvp_maxt

'''

#trajectory = first_trajectory  #+ rank  NOTE: uncomment "+ rank" when parallelizing

print('computing time-evolution for trajectory {}'.format(trajectory) )


#COMPUTE ONE TRAJECTORY WITH TDVP AND ADAPTIVE TIMESTEP
#test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(init_state, conf_tdvp, tdvp_maxt, tdvp_dt, tol, max_iterations, trajectory, obsdict, tdvp_trunc_threshold, tdvp_trunc_weight, tdvp_trunc_maxStates)

for trajectory in range(n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(init_state, conf_tdvp, tdvp_maxt, tdvp_dt, tol, max_iterations, trajectory, obsdict, tdvp_trunc_threshold, tdvp_trunc_weight, tdvp_trunc_maxStates)

read_directory = os.getcwd()
write_directory = os.getcwd()


obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 

'''

trajectory = first_trajectory  #+ rank  NOTE: uncomment "+ rank" when parallelizing
print('computing time-evolution for trajectory {}'.format(trajectory) )

#COMPUTE ONE TRAJECTORY WITH TDVP AND ADAPTIVE TIMESTEP
#test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(init_state, conf_tdvp, tdvp_maxt, tdvp_dt, tol, max_iterations, trajectory, obsdict, tdvp_trunc_threshold, tdvp_trunc_weight, tdvp_trunc_maxStates)


for trajectory in range(first_trajectory, n_trajectories + first_trajectory): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(init_state, conf_tdvp, tdvp_maxt, tdvp_dt, trajectory, obsdict)


read_directory = os.getcwd()
write_directory = os.getcwd()


#obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 

