import numpy as np
import os
from evos.src.lattice.bos_spin_one_half_lattice import BosSpinOneHalfLattice
from evos.src.methods.ed_quantum_jumps_parallel import EdQuantumJumps
from evos.src.observables.observables_parallel import ObservablesDict 
import sys

# sites of the lattic 0 for bosons, 1 for spin 1/2
sites = [0, 1, 1, 0]
# local dimension of the boson sites
bos_dim = 3
# inverse temperature
beta = 5.
# energy scale of the boson sites
omega_b = 5.
# energy scale of the spin 1/2 sites
omega_s = 2.5
# time until which the time evolution is carried out
time = 50.
# time steps
delta_t = 0.1
# number of steps for the time evolution
n_timesteps = int(time/delta_t)
# coupling strength between the baths and boson sites
k1 = 25.
# coupling strength between the boson and spin 1/2 sites
k2 = 2.
# coupling strength between the spin sites
k_x = 0.1
k_y = 0.
k_z = 0.

try:
    os.mkdir('data_qj')
except:
    pass

try:
    os.chdir('data_qj')
except:
	pass

lat = BosSpinOneHalfLattice(sites=sites, bos_dim=bos_dim)

dim_ges = lat.dim_ges
H = np.zeros((dim_ges, dim_ges), dtype='complex')
for i in range(lat.n_sites):
    if lat.sites[i]:
        # add local terms for the spin 1/2 sites
        H += omega_s * lat.sso('sz', i)
    else:
        # add local terms for the boson sites
        H += omega_b * lat.sso('n_bos', i)
for i in range(lat.n_sites-1):
    if lat.sites[i] and lat.sites[i+1]:
        # add interaction between neighboring spin 1/2 sites
        H += k_x * np.matmul(lat.sso('sx', i), lat.sso('sx', i+1))
        H += k_y * np.matmul(lat.sso('sy', i), lat.sso('sy', i+1))
        H += k_z * np.matmul(lat.sso('sz', i), lat.sso('sz', i+1))
    elif lat.sites[i+1]:
        # add interaction between neighboring boson and spin 1/2 site
        H += k2 * (np.matmul(lat.sso('bp', i), lat.sso('sm', i+1))+np.matmul(lat.sso('bm', i), lat.sso('sp', i+1)))
    elif lat.site[i]:
        # add interaction between neighboring spin 1/2 and boson site
        H += k2 * (np.matmul(lat.sso('sm', i), lat.sso('bp', i+1))+np.matmul(lat.sso('sp', i), lat.sso('bm', i+1)))

alpha = np.exp(- beta * omega_b)
# create Lindbladians
L = []
for i in range(lat.n_sites):
    if not lat.sites[i]:
        L.append(np.sqrt(alpha*k1) * lat.sso("bp", i))
        L.append(np.sqrt(k1) * lat.sso("bm", i))

ed_quantum_jumps = EdQuantumJumps(lat.n_sites, H, L)

# initialize dictionary for saving observables
obsdict = ObservablesDict()
obsdict.initialize_observable('sz_0', (1,), n_timesteps)
obsdict.initialize_observable('sz_1', (1,), n_timesteps)

sz_0 = lat.sso('sz', 1)
sz_1 = lat.sso('sz', 2)


def compute_sz_0(state, obs_array_shape, dtype):
    obs_array = np.zeros(obs_array_shape, dtype=dtype)
    # OBS DEPENDENT PART START
    obs_array[0] = np.real(np.dot(np.dot(np.conjugate(state), sz_0), state))
    # OBS DEPENDENT PART END
    return obs_array


def compute_sz_1(state, obs_array_shape, dtype):
    obs_array = np.zeros(obs_array_shape, dtype=dtype)
    # OBS DEPENDENT PART START
    obs_array[0] = np.real(np.dot(np.dot(np.conjugate(state), sz_1), state))
    # OBS DEPENDENT PART END
    return obs_array


obsdict.add_observable_computing_function('sz_0', compute_sz_0)
obsdict.add_observable_computing_function('sz_1', compute_sz_1)

# compute qj trajectories sequentially
# for trajectory in range(n_trajectories): #FIXME: looping over trajectories seems to be problematic!
trajectory = int(sys.argv[1])
test_singlet_traj_evolution = ed_quantum_jumps.quantum_jump_single_trajectory_time_evolution(lat.vacuum_state, time, delta_t, trajectory, obsdict)

# trajectory = 1
# test_singlet_traj_evolution = ed_quantum_jumps.quantum_jump_single_trajectory_time_evolution(lat.vacuum_state, time, delta_t, trajectory, obsdict)
# averages and errors
# read_directory = os.getcwd()
# write_directory = os.getcwd()

# n_trajectories = 2
# obsdict.compute_trajectories_averages_and_errors(n_trajectories, read_directory, write_directory, remove_single_trajectories_results=False)

