import evos
import evos.src.lattice.lattice as lat
import evos.src.methods.lindblad as lindblad
import numpy as np
import matplotlib.pyplot as plt
n_sites=1

spin_lat_test = lat.Lattice('ed')
spin_lat_test.specify_lattice('spin_one_half_lattice')
spin_lat_test = spin_lat_test.spin_one_half_lattice.SpinOneHalfLattice(n_sites)
sx_0 = spin_lat_test.sso('sx',0)
sy_0 = spin_lat_test.sso('sy',0)
sz_0 = spin_lat_test.sso('sz',0)

w = 1.
lam = 1.
H = w/2. * sz_0
L =  lam/2. * np.matrix(sz_0) #Lindblad operators must be cast from arrays to matrices in order to be able to use .H
dt = 0.1
t_max = 20
n_timesteps = int(t_max/dt)

#vac = spin_lat_test.vacuum_state #vacuum state = all up
#down = np.dot( sx_0, vac.copy())
#init_state = 1./np.sqrt(2.) * (vac + down)
#init_projector = lindblad_test.ket_to_projector(init_state)

#initial state
rx = np.sqrt(1/5)
ry = np.sqrt(1/5)
rz = np.sqrt(3/5)

def density_matrix_from_bloch_vector(rx, ry, rz):
    return 0.5* np.matrix( [ [1+rz, rx-1j*ry], [rx+1j*ry, 1-rz] ],dtype=complex )

rho_0 = density_matrix_from_bloch_vector(rx, ry, rz)
# print(rho_0)
# quit()
lindblad_test = evos.src.methods.lindblad.Lindblad([L],H,n_sites)

rho_t = lindblad_test.solve_lindblad_equation(rho_0, dt, t_max)

names_and_operators_list = {'sx_0': sx_0, 'sy_0': sy_0, 'sz_0': sz_0}
obs_test_dict =  lindblad_test.compute_observables(rho_t, names_and_operators_list, dt, t_max )
sx_0_test = obs_test_dict['sx_0']
sy_0_test = obs_test_dict['sy_0']
sz_0_test = obs_test_dict['sz_0']

# #compute observables
exp_sx0 = np.zeros(n_timesteps)
exp_sy0 = np.zeros(n_timesteps)
exp_sz0 = np.zeros(n_timesteps)
for i in range( n_timesteps ):
    prod_sx0 = np.matmul(sx_0,rho_t[:,:,i])
    prod_sy0 = np.matmul(sy_0,rho_t[:,:,i])
    prod_sz0 = np.matmul(sz_0,rho_t[:,:,i])
    
    exp_sx0[i] = np.trace(prod_sx0)
    exp_sy0[i] = np.trace(prod_sy0)
    exp_sz0[i] = np.trace(prod_sz0)
    
    
# #PLOT 
time_v = np.linspace(0, t_max, n_timesteps )

plt.plot(time_v, exp_sx0, label = 'sx')    
plt.plot(time_v, exp_sy0, label = 'sy')    
plt.plot(time_v, exp_sz0, label = 'sz')  
  
plt.plot(time_v, sx_0_test, label = 'sx_test')    
plt.plot(time_v, sy_0_test, label = 'sy_test')    
plt.plot(time_v, sz_0_test, label = 'sz_test')    
plt.legend()
plt.show()