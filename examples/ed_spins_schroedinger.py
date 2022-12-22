import evos
import evos.src.lattice as lat
#import evos.src.methods.lindblad as lindblad
import evos.src.methods.ed_schroedinger as ed_schroedinger
import evos.src.observables.observables as observables
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import scipy.linalg as la

time_start = time.process_time()

#parameters
n_sites = 2
dim_H = 2 ** n_sites
J = 1
gamma = 1
W = 10
seed = 1
np.random.seed(seed)
eps_vec = np.random.uniform(0, W, n_sites)
dt = 0.01
t_max = 10
n_timesteps = int(t_max/dt)
n_trajectories = 1
trajectory = 0 

#os.chdir('benchmark')
try:
    os.system('mkdir data_schroedinger')
    os.chdir('data_schroedinger')
except:
    pass

#lattice
time_lat = time.process_time()
spin_lat = lat.Lattice('ed')
spin_lat.specify_lattice('spin_one_half_lattice')
spin_lat = spin_lat.spin_one_half_lattice.SpinOneHalfLattice(n_sites)
#np.save( 'time_lat_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lat-time.process_time())
print('time_lat:{0}'.format( time.process_time() - time_lat ) )

#Hamiltonian
time_H = time.process_time()
H = np.zeros( (dim_H,dim_H), dtype=complex)
#spin coupling
for i in range(n_sites):
    for j in range(n_sites):
        if j != i: 
            H += J/np.abs(i-j)**3 * ( np.matmul( spin_lat.sso('sp',i),spin_lat.sso('sm',j) )   + np.matmul( spin_lat.sso('sp',j),spin_lat.sso('sm',i) ) )
#disorder
for i in range(n_sites):
    H += eps_vec[i] * spin_lat.sso('sz',i)

assert (np.matrix(H).H).all() == (np.matrix(H)).all()      #check wheter Hamiltonian symmetric    
#np.save( 'time_H_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_H-time.process_time())
print('time_H:{0}'.format( time.process_time()- time_H ) )

#initial state: antiferromagnetic
init_state = spin_lat.vacuum_state #vacuum state = all up
for i in np.arange(1,n_sites,2): #flip every second spin down
    init_state = np.dot(spin_lat.sso('sx',i), init_state.copy() ) 
init_state /= la.norm(init_state)

# print('LA.norm(init_state) :', la.norm(init_state))

#observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('sz_0',(1,), n_timesteps) #1D
obsdict.initialize_observable('sz_1',(1,), n_timesteps) #1D

sz_0 = spin_lat.sso( 'sz', 0 )
sz_1 = spin_lat.sso( 'sz', 1 )
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
obsdict.add_observable_computing_function('sz_1',compute_sz_1 )


#compute schroedinger time-evolution
ed_schroedinger = EdSchroedinger(n_sites, H)
ed_schroedinger.schroedinger_time_evolution(self, init_state, t_max, dt, obsdict)

print('process time: ', time.process_time() - time_start )