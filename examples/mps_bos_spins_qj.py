import numpy as np
import os
from evos.src.methods.mps_quantum_jumps_no_normalization_adaptive_timestep_parallel import MPSQuantumJumps
from evos.src.observables.observables_parallel import ObservablesDict 
import sys
import pyten as p

size = 4
# spin-S degrees of freedom
S = 0.5
# sites of the lattic 1 for bosons, 0 for spin-S
sites = [i % 2 for i in range(2*size)]
ca_ops = ["a", "ah"]
# local dimension of the boson sites
dim_bos = 7
# inverse temperature
beta = 20.
# energy scale of the boson sites
omega_b = 5.
# energy scale of the spin 1/2 sites
omega_s = 2.5
# coupling strength between the baths and boson sites
k1 = 25.
# coupling strength between the boson and spin 1/2 sites
k2 = 2.
# coupling strength between the spin sites
k_x = 0.
k_y = 0.
k_z = 0.1

# paramters for time evolution
conf_tdvp = p.tdvp.Conf()
conf_tdvp.maxt = 15.
conf_tdvp.mode = p.tdvp.Mode.TwoSite
conf_tdvp.dt = 0.005
conf_tdvp.trunc.threshold = 1e-8
conf_tdvp.trunc.weight = 1e-10
conf_tdvp.trunc.maxStates = 500
conf_tdvp.exp_conf.errTolerance = 1e-7
conf_tdvp.exp_conf.inxTolerance = 1e-6
conf_tdvp.exp_conf.maxIter = 10
conf_tdvp.cache = 1

n_timesteps = int(np.real(conf_tdvp.maxt/conf_tdvp.dt))

# parameters for bisection method
tol = 1e-4
max_iter = 10

try:
    with open("parameters.txt", "x") as f:
        params = [
            "size = {}\n".format(size),
            "S = {}\n".format(S),
            "dim_bos = {}\n".format(dim_bos),
            "beta = {}\n".format(beta),
            "omega_b = {}\n".format(omega_b),
            "omega_s = {}\n".format(omega_s),
            "k_1 = {}\n".format(k1),
            "k_2 = {}\n".format(k2),
            "k_x = {}\n".format(k_x),
            "k_y = {}\n".format(k_y),
            "k_z = {}\n\n".format(k_z),
            "conf_tdvp:\n"
            "maxt = {}\n".format(np.real(conf_tdvp.maxt)),
            "dt = {}\n".format(np.real(conf_tdvp.dt)),
            "mode = TwoSite\n",
            "cache = {}\n".format(np.real(conf_tdvp.cache)),
            "trunc.threshold = {}\n".format(np.real(conf_tdvp.trunc.threshold)),
            "trunc.weight = {}\n".format(np.real(conf_tdvp.trunc.weight)),
            "trunc.maxStates = {}\n".format(np.real(conf_tdvp.trunc.maxStates)),
            "exp_conf.errTolerance = {}\n".format(np.real(conf_tdvp.exp_conf.errTolerance)),
            "exp_conf.inxTolerance = {}\n".format(np.real(conf_tdvp.exp_conf.inxTolerance)),
            "exp_conf.maxIter = {}\n\n".format(np.real(conf_tdvp.exp_conf.maxIter)),
            "bisection_params:\n",
            "tol = {}\n".format(tol),
            "max_iterations = {}".format(max_iter)
        ]
        f.writelines(params)
except:
    pass

try:
    os.mkdir("data_qj")
except:
    pass

try:
    os.chdir("data_qj")
except:
	pass

lat = p.mp.lat.u1u1.genSpinBosonLattice(sites, S, dim_bos)
# second parameter denotes the sublattice label which is to be purified
pp_lat = p.mp.proj_pur.proj_purification(lat, [1], ca_ops)

# generate state with S=0, no bosons in phys. system and dim_bos bosons in bath system
initial_state = p.mp.proj_pur.generateNearVacuumState(pp_lat, 2, "0,0,{}".format(dim_bos))

H = []
for i in range(size):
    # add local terms for boson sites
    H.append(omega_b * pp_lat.get("nb", 3*i + 1))
    # add local terms for spin sites
    H.append(omega_s * pp_lat.get("sz", 3*i))
    # boson annihilation/creation operators with balancing operators (to preserve U1)
    bm_i = pp_lat.get("a", 3*i + 1) * pp_lat.get("ah", 3*i + 2)
    bp_i = pp_lat.get("ah", 3*i + 1) * pp_lat.get("a", 3*i + 2)
    # add interaction between spin and boson sites
    H.append(k2 * (pp_lat.get("s+", 3*i) * bm_i + pp_lat.get("s-", 3*i) * bp_i))
for i in range(size-1):
    # add interaction between spin sites (only z-component for now)
    H.append(k_z * pp_lat.get("sz", 3*i) * pp_lat.get("sz", 3*(i+1)))

H = p.mp.addLog(H)

alpha = np.exp(- beta * omega_b)
# create Lindbladians
L = []
for i in range(size):
    bm_i = pp_lat.get("a", 3*i + 1) * pp_lat.get("ah", 3*i + 2)
    bp_i = pp_lat.get("ah", 3*i + 1) * pp_lat.get("a", 3*i + 2)
    L.append(np.sqrt(alpha*k1) * bp_i)
    L.append(np.sqrt(k1) * bm_i)

mps_quantum_jumps = MPSQuantumJumps(3*size, pp_lat, H, L)

# initialize dictionary for saving observables
obsdict = ObservablesDict()
obsdict.initialize_observable("sz", (size,), n_timesteps)
obsdict.initialize_observable("E", (1,), n_timesteps)
obsdict.initialize_observable("SS_Corr", (size-1,), n_timesteps)
obsdict.initialize_observable("SB_Corr", (size,), n_timesteps)
obsdict.initialize_observable("bdim_mat", (3*size,), n_timesteps)

def compute_sz(state, obs_array_shape, dtype):
    obs_array = np.zeros(obs_array_shape, dtype=dtype)
    # OBS DEPENDENT PART START
    for i in range(size):
        obs_array[i] = np.real(p.mp.expectation(state, pp_lat.get("sz", 3*i)))
    # OBS DEPENDENT PART END
    return obs_array


def compute_E(state, obs_array_shape, dtype):
    obs_array = np.zeros(obs_array_shape, dtype=dtype)
    # OBS DEPENDENT PART START
    for i in range(size):
        obs_array[0] += np.real(p.mp.expectation(state, pp_lat.get("sz", 3*i)))
    # OBS DEPENDENT PART END
    return obs_array


def compute_SS_Corr(state, obs_array_shape, dtype):
    obs_array = np.zeros(obs_array_shape, dtype=dtype)
    for i in range(size-1):
        SzSz = p.mp.expectation(state, pp_lat.get("sz", 3*i) * pp_lat.get("sz", 3*(i+1)))
        Sz_0 = p.mp.expectation(state, pp_lat.get("sz", 3*i))
        Sz_1 = p.mp.expectation(state, pp_lat.get("sz", 3*(i+1)))
        obs_array[i] = np.real(SzSz - Sz_0*Sz_1)
    return obs_array


def compute_SB_Corr(state, obs_array_shape, dtype):
    obs_array = np.zeros(obs_array_shape, dtype=dtype)
    for i in range(size):
        SzNb = p.mp.expectation(state, pp_lat.get("sz", 3*i) * pp_lat.get("nb", 3*i+1))
        Sz = p.mp.expectation(state, pp_lat.get("sz", 3*i))
        Nb = p.mp.expectation(state, pp_lat.get("nb", 3*i+1))
        obs_array[i] = np.real(SzNb - Sz*Nb)
    return obs_array


def compute_bdim_mat(state, obs_array_shape, dtype):
    obs_array = np.zeros(obs_array_shape, dtype=dtype)
    for i in range(3*size):
        obs_array[i] = state[i].getTotalDims()[1] / state.norm() ** 2
    return obs_array

obsdict.add_observable_computing_function("sz", compute_sz)
obsdict.add_observable_computing_function("E", compute_E)
obsdict.add_observable_computing_function("SS_Corr", compute_SS_Corr)
obsdict.add_observable_computing_function("SB_Corr", compute_SB_Corr)
obsdict.add_observable_computing_function("bdim_mat", compute_bdim_mat)

# compute qj trajectories sequentially
# for trajectory in range(n_trajectories): #FIXME: looping over trajectories seems to be problematic!
trajectory = int(sys.argv[1])
test_singlet_traj_evolution = mps_quantum_jumps.quantum_jump_single_trajectory_time_evolution(initial_state, conf_tdvp, np.real(conf_tdvp.maxt), np.real(conf_tdvp.dt), tol, max_iter, trajectory, obsdict, conf_tdvp.trunc.threshold, conf_tdvp.trunc.weight, conf_tdvp.trunc.maxStates)

