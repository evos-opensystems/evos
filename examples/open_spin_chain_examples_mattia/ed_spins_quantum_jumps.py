import evos
import evos.src.lattice.spin_one_half_lattice as lat
#import evos.src.methods.lindblad as lindblad
import evos.src.methods.ed_quantum_jumps as ed_quantum_jumps
import evos.src.observables.observables as observables
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import scipy.linalg as la
time_start = time.process_time()
#DO BENCHMARK OF TEVO AND OBSERVABLES!
#parameters
n_sites = 2
dim_H = 2 ** n_sites
J = 1
gamma = 1
W = 10
seed_W = 7
rng = np.random.default_rng(seed=seed_W) # random numbers
eps_vec = rng.uniform(0, W, n_sites) #onsite disordered energy random numbers
dt = 0.01
t_max = 10
n_timesteps = int(t_max/dt)
n_trajectories = 1
trajectory = 0 


#os.chdir('benchmark')
try:
    os.system('mkdir data_qj_ed')
    os.chdir('data_qj_ed')
except:
    pass

try:
    shutil.rmtree('0')
    shutil.rmtree('1')
except:
    pass

#lattice
time_lat = time.process_time()
#spin_lat = lat.Lattice('ed')
#spin_lat.specify_lattice('spin_one_half_lattice')
spin_lat = lat.SpinOneHalfLattice(n_sites)
#np.save( 'time_lat_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lat-time.process_time())
#print('time_lat:{0}'.format( time.process_time() - time_lat ) )

#Hamiltonian
time_H = time.process_time()
H = np.zeros( (dim_H,dim_H), dtype=complex)
#spin coupling
for i in range(n_sites):
    for j in range(i):
        H += J/np.abs(i-j)**3 * ( np.matmul( spin_lat.sso('sp',i),spin_lat.sso('sm',j) )   + np.matmul( spin_lat.sso('sp',j),spin_lat.sso('sm',i) ) )
#disorder
for i in range(n_sites):
    H += eps_vec[i] * spin_lat.sso('sz',i)

assert (np.matrix(H).H).all() == (np.matrix(H)).all()      #check wheter Hamiltonian symmetric    
#np.save( 'time_H_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_H-time.process_time())
#print('time_H:{0}'.format( time.process_time()- time_H ) )

#initial state: antiferromagnetic
init_state = spin_lat.vacuum_state #vacuum state = all up
for i in np.arange(1,n_sites,2): #flip every second spin down
    init_state = np.dot(spin_lat.sso('sx',i), init_state.copy() ) 
init_state /= la.norm(init_state)

#print(init_state)

# print('LA.norm(init_state) :', la.norm(init_state))

#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('sz_0',(1,), n_timesteps) #1D
#obsdict.initialize_observable('sz_1',(1,), n_timesteps) #1D


sz_0 = spin_lat.sso( 'sz', 0 )
sz_1 = spin_lat.sso( 'sz', 1 )

print(np.shape(sz_0))
print(np.shape(init_state))
###
# sz_0_init_state = np.dot( np.conjugate(init_state), np.dot(sz_0,init_state ))
# print(sz_0_init_state)
###

def compute_sz_0(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.dot( np.dot( np.conjugate(state),sz_0 ), state )  )  
    #OBS DEPENDENT PART END
    return obs_array

def compute_sz_1(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array[0] = np.real( np.dot( np.dot( np.conjugate(state),sz_1 ), state )  )
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('sz_0',compute_sz_0 )
#obsdict.add_observable_computing_function('sz_1',compute_sz_1 )



#Lindbladian: dissipation only on central site
L = gamma * spin_lat.sso( 'sm', 1 ) #int( n_sites/2 )


ed_quantum_jumps = ed_quantum_jumps.EdQuantumJumps(n_sites, H, [L])

#compute qj trajectories sequentially
for trajectory in range(n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = ed_quantum_jumps.quantum_jump_single_trajectory_time_evolution(init_state, t_max, dt, trajectory, obsdict )

#averages and errors
read_directory = os.getcwd()
write_directory = os.getcwd()


obsdict.compute_trajectories_averages_and_errors( list(range(n_trajectories)), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 


print('process time: ', time.process_time() - time_start )

