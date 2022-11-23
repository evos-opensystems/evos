import evos
import evos.src.lattice as lat
import evos.src.methods.lindblad as lindblad
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.linalg as la
#import spinful_fermions_lattice as spinful_fermions_lattice
import evos.src.lattice.spin_one_half_lattice as spin_one_half_lattice
import evos.src.lattice.spinful_fermions_lattice

#DO BENCHMARK OF TEVO AND OBSERVABLES!
#parameters
n_sites = 2
dim_H = 2 ** n_sites
J = 1
gamma=1
W = 10
seed = 1
np.random.seed(seed)
eps_vec = np.random.uniform(0, W, n_sites)
dt = 0.01
t_max = 10
n_timesteps = int(t_max/dt)

#os.chdir('benchmark')
#lattice
time_lat = time.process_time()
spin_lat = lat.Lattice('ed')
spin_lat.specify_lattice('spinful_fermions_lattice')


#spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_sites)


#np.save( 'time_lat_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lat-time.process_time())
print('time_lat:{0}'.format( time.process_time() - time_lat ) )


'''
#Hamiltonian
time_H = time.process_time()
def H(J, U, V): 
        
    hop = np.zeros((dim_H, dim_H))
    for k in range(1, n_sites): 
        
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k +1,N), c_down_dag(k,N))
        hop += np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) 
        + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down'))
     
    
        
    Coul = np.zeros((dim_H, dim_H))
    for k in range(1, n_sites+1): 
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
        
        #print(Sx_nn_new)
        Coul += np.dot(n_up,n_down)
        
    Coul_nn = np.zeros((dim_H, dim_H))
    for k in range(1, n_sites+1):
        
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
        
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        Coul_nn += np.dot( ,spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
        
    
        
    H = - J* hop + U* Coul
    return H

print('time_H:{0}'.format( time.process_time()- time_H ) )

#initial state: antiferromagnetic
init_state = spin_lat.vacuum_state #vacuum state = all up
for i in np.arange(1,n_sites,2): #flip every second spin down
    init_state = np.dot( spin_lat.sso('sx',i), init_state.copy() )

init_state /= la.norm(init_state)

#Lindbladian: dissipation only on central site
L = gamma * np.matrix( spin_lat.sso( 'sm', 0 ) )  # int( n_sites/2 ) #Lindblad operators must be cast from arrays to matrices in order to be able to use .H
time_lind_evo = time.process_time()
lindblad = evos.src.methods.lindblad.Lindblad([L],H,n_sites)
rho_0 = lindblad.ket_to_projector(init_state)        
rho_t = lindblad.solve_lindblad_equation(rho_0, dt, t_max)
#np.save( 'time_lind_evo_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lind_evo-time.process_time())
print('time_lind_evo:{0}'.format( time.process_time() - time_lind_evo ) )
#observables
time_lind_obs = time.process_time()
names_and_operators_list = {} #{'sz_0': spin_lat.sso('sz',0), 'sz_1': spin_lat.sso('sz',1), 'sz_2': , 'sz_3': sz_3 }
for i in range(n_sites):
    names_and_operators_list.update({'sz_'+str(i) : spin_lat.sso('sz',i) })
obs_test_dict =  lindblad.compute_observables(rho_t, names_and_operators_list, dt, t_max )
#np.save( 'time_lind_obs_n_sites' +str(n_sites) + '_n_timesteps' + str(n_timesteps), time_lind_obs-time.process_time())
print('time_lind_obs:{0}'.format( time.process_time() - time_lind_evo ) )
#PLOT
time_v = np.linspace(0, t_max, n_timesteps )

for i in range(n_sites):
    plt.plot(time_v, obs_test_dict['sz_'+str(i)], label = '<sz> on site '+str(i))    

plt.legend()
plt.xlabel('time')
plt.show()

#matrix plot
print(len(obs_test_dict))
sz_matrix = np.zeros( ( len( obs_test_dict ) , n_timesteps ) )
for i in range(n_sites):
    sz_matrix[i,:] = obs_test_dict['sz_'+str(i)]

try:
    os.system('mkdir data_lindblad')
    os.chdir('data_lindblad')
except:
    pass    
np.savetxt('sz_matrix_n_sites' +str(n_sites) + '_t_max' + str(t_max) + '_dt' +str(dt) , sz_matrix )

# plt.imshow(sz_matrix, aspect='auto', extent=[0,t_max,1,n_sites])
# plt.yticks(range(1,n_sites+1))

# plt.colorbar()
# plt.xlabel(r'time $(1/\bar{J})$' )
# plt.ylabel('sites')
# plt.title('Lindblad evolution of the magnetization '+r'$\langle \hat{S_z} \rangle $ ' +'starting from the Neel state' +'\n Dissipation on central site c: ' + r'$\hat{L} = \gamma_c \hat{S}^-_c$ with $\gamma_c = 0.2 \bar{J}$. Disorder strength $W= 0.2 \bar{J}$ ')
# plt.show()    
'''