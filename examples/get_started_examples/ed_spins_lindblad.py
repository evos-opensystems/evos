"""Solving the Lindblad equation with exact diagonalization (ED) for a spin chain with algebraically decaying interactions, disorder and dissipation on one site."""

import evos
import evos.src.lattice.spin_one_half_lattice as spin_lat
import evos.src.methods.lindblad as lindblad
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.linalg as la

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
dt = 0.005
t_max = 5
n_timesteps = int(t_max/dt)

#lattice
time_lat = time.process_time()
spin_lat = spin_lat.SpinOneHalfLattice(n_sites)
print('time_lat:{0}'.format( time.process_time() - time_lat ) )

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
print('time_H:{0}'.format( time.process_time()- time_H ) )

#initial state: antiferromagnetic
init_state = spin_lat.vacuum_state #vacuum state = all up
for i in np.arange(1,n_sites,2): #flip every second spin down
    init_state = np.dot( spin_lat.sso('sx',i), init_state.copy() )

init_state /= la.norm(init_state)

#Lindbladian: dissipation only on central site
L = gamma * spin_lat.sso( 'sm', int( n_sites/2 ) ) # int( n_sites/2 ) #Lindblad operators must be cast from arrays to matrices in order to be able to use .H
time_lind_evo = time.process_time()


lindblad = lindblad.Lindblad(n_sites)
rho_0 = lindblad.ket_to_projector(init_state)        
rho_t = lindblad.solve_lindblad_equation(rho_0, dt, t_max, [L], H)
#np.save( 'time_lind_evo_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lind_evo-time.process_time())
print('time_lind_evo:{0}'.format( time.process_time() - time_lind_evo ) )
#observables
time_lind_obs = time.process_time()
names_and_operators_list = {} #{'sz_0': spin_lat.sso('sz',0), 'sz_1': spin_lat.sso('sz',1), 'sz_2': , 'sz_3': sz_3 }
for i in range(n_sites):
    names_and_operators_list.update({'sz_'+str(i) : spin_lat.sso('sz',i) })
obs_test_dict =  lindblad.compute_observables(rho_t, names_and_operators_list, dt, t_max)
#np.save( 'time_lind_obs_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lind_obs-time.process_time())
print('time_lind_obs:{0}'.format( time.process_time() - time_lind_evo ) )


#save sz
sz_matrix = np.zeros( ( len( obs_test_dict ) , n_timesteps ) )
for i in range(n_sites):
    sz_matrix[i,:] = obs_test_dict['sz_'+str(i)]
try:
    os.system('mkdir data_lindblad')
    os.chdir('data_lindblad')
except:
    pass    

np.savetxt('sz_av', sz_matrix)

#PLOT
fig, axs = plt.subplots()
time_v = np.linspace(0, t_max, n_timesteps )

for i in range(n_sites):
    plt.plot(time_v, obs_test_dict['sz_'+str(i)], label = '<sz> on site '+str(i))    

plt.legend()
plt.xlabel('time')
fig.savefig('ed_spins_lindblad.png')

