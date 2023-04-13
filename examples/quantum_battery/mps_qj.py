"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with mps quantum jumps. The dimension of the oscillator needs to be strongly truncated.
"""
import evos.src.methods.mps_quantum_jumps as mps_quantum_jumps
import evos.src.observables.observables_pickled as observables
import pyten as ptn
import numpy as np 
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
#import math
import os
from scipy import linalg as sla
sys.stdout.write('test')
import argparse

arg_parser = argparse.ArgumentParser(description = "Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with ed lindblad. The dimension of the oscillator needs to be strongly truncated.")
arg_parser.add_argument("-b",   "--bosons", dest = 'max_bosons',  default = 4, type = int, help = 'number of bosonic degrees of freedom - 1 [4]')
arg_parser.add_argument("-dt",   "--timestep", dest = 'dt',  default = 0.02, type = float, help = 'timestep [0.02]')
arg_parser.add_argument("-t_max",   "--max_time", dest = 't_max',  default = 5, type = float, help = 'maximal simulated time [5]')
arg_parser.add_argument("-mu_l",   "--checmical_pot_left_lead", dest = 'mu_l',  default = +0.5, type = float, help = 'checmical pot. left lead [0.5]')
arg_parser.add_argument("-mu_r",   "--checmical_pot_right_lead", dest = 'mu_r',  default = -0.5, type = float, help = 'checmical pot. right lead [-0.5]')

#FIXME: ADD MU_L AND MU_R
args = arg_parser.parse_args()


np.set_printoptions(threshold=sys.maxsize)
sys.stdout.write('test')

#PARAMETERS
max_bosons = args.max_bosons

om_0 = 0.2
m = 1
lamb = 0.1
x0 = np.sqrt( 2./ (m * om_0) )
F = 2 *lamb / x0

eps = 0  
Om_kl = +0.5
Om_kr = -0.5
Gamma = 2
g_kl = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
g_kr = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
N0 = 0.5 #FIXME: is this correct?
delta_l = 1
delta_r = 1

mu_l = args.mu_l
mu_r = args.mu_r

T_l = 1./0.5 #beta_l = 0.5
T_r = 1./0.5 #beta_r = 0.5
k_b = 1 #boltzmann constant
 
dt = args.dt
t_max = args.t_max
time_v = np.arange(0, t_max, dt)
n_timesteps = int(t_max/dt)
n_trajectories = 1
first_trajectory = 0

#Lattice
ferm_bos_sites = [ 1, 1, 1, 1, 0, 1, 1 ] #doubled the fermionic sites to project-purify
lat = ptn.mp.lat.u1u1.genSpinlessFermiBose(ferm_bos_sites, max_bosons)
lat = ptn.mp.proj_pur.proj_purification(lat, [0], ["a", "ah"])

vac_state =  ptn.mp.proj_pur.generateNearVacuumState(lat, 2, "0," + str( max_bosons ) )
#vac_state = ptn.mp.generateNearVacuumState(lat)

#prepare PP vacuum for fermions
vac_state *= lat.get('ch',1)    
vac_state *= lat.get('ch',3) 
vac_state *= lat.get('ch',7)    

#FIXME: exite one particle in the left lead (USED TO DEBUGG WITHOUT INJECTION)
vac_state *= lat.get('ch',0) * lat.get('c',1) 
#vac_state *= lat.get('ch',6) * lat.get('c',7)

################################################################
# for site in range(8):
#     print(site, ptn.mp.expectation(vac_state, lat.get('n',site)))
# quit()    
################################################################
   
class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps): #system
        h_s = eps * lat.get('c',2) * lat.get('ch',2)  #no need to PP
        #h_s = ptn.mp.addLog(h_s)
        #h_s.truncate()
        return h_s 

    def h_b(self, Om_kl, Om_kr): #leads
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = []
        h_b.append( Om_kl * lat.get('c',0) * lat.get('ch',0) ) #no need to PP
        h_b.append( Om_kr * lat.get('c',6) * lat.get('ch',6) ) #no need to PP
        h_b = ptn.mp.addLog(h_b)
        #h_b.truncate()
        return h_b
   
    def h_t(self, g_kl, g_kr): #system-leads
        
        h_t = g_kl * ( lat.get('c',0) * lat.get('ch',2)  +  lat.get('c',2) * lat.get('ch',0) ) 
        h_t += g_kr * (lat.get('c',6)  * lat.get('ch',2)  + lat.get('c',2) * lat.get('ch',6) ) 
    
        #h_t.truncate()
        return h_t
    
    def h_boson(self, om_0): #oscillator
        h_boson = om_0 * lat.get('nb',4) 
        return h_boson
    
    def h_v(self, F, N0): #system-oscillator
        h_v = - F * ( lat.get('nf',2) - N0 * lat.get('I') ) *  ( lat.get('ah',4) * lat.get('a',5)  + lat.get('a',4) * lat.get('ah',5) ) 
        #h_v.truncate()
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F):
        h_tot =  self.h_boson(om_0) + self.h_v(F, N0) + self.h_s(eps) + self.h_t(g_kl, g_kr) + self.h_b(Om_kl, Om_kr) 
        #h_tot.truncate()
        return h_tot
        
    
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)
h_s = ham.h_s(eps)
#h_b = ham.h_b(Om_kl, Om_kr, mu_l, mu_r)
# h_t = ham.h_t(g_kl, g_kr)
h_boson = ham.h_boson(om_0)
# h_v = ham.h_v(om_0, F)
h_tot = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F)

# lat.add('h_tot', 'h_tot', h_tot)
# lat.save('lat')
# quit()

########### GS
# conf = ptn.dmrg.DMRGConfig()

# # give us a list to add stages
# stages = []

# #first stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[0].trunc.maxStates = 16
# stages[0].convergenceMaxSweeps = 200
# stages[0].trunc.weight = 1e-6
# stages[0].trunc.threshold = 1e-8
# stages[0].convergenceMinSweeps = 50
# #stages[0].convMinEnergyDiff = -1
# stages[0].mode.DMRG3S
# #second stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[1].trunc.maxStates = 32
# stages[1].convergenceMaxSweeps = 150
# stages[1].trunc.weight = 1e-7
# stages[1].trunc.threshold = 1e-9
# stages[1].convergenceMinSweeps = 40
# #stages[1].convMinEnergyDiff = -1
# stages[1].mode.DMRG3S

# #third stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[2].trunc.maxStates = 64
# stages[2].convergenceMaxSweeps = 100
# stages[2].trunc.weight = 1e-8
# stages[2].trunc.threshold = 1e-10
# stages[2].convergenceMinSweeps = 30
# #[2].convMinEnergyDiff = -1
# stages[2].mode.TwoSite

# #fourth stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[3].trunc.maxStates = 128
# stages[3].convergenceMaxSweeps = 100
# stages[3].trunc.weight = 1e-10
# stages[3].trunc.threshold = 1e-12
# stages[3].convergenceMinSweeps = 25
# #stages[3].convMinEnergyDiff = -1
# stages[3].mode.DMRG3S

# #fifth stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[4].trunc.maxStates = 256
# stages[4].convergenceMaxSweeps = 100
# stages[4].trunc.weight = 1e-11
# stages[4].trunc.threshold = 1e-13
# stages[4].convergenceMinSweeps = 20
# #stages[4].convMinEnergyDiff = -1
# stages[4].mode.DMRG3S

# #6th stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[5].trunc.maxStates = 512
# stages[5].convergenceMaxSweeps = 100
# stages[5].trunc.weight = 1e-13
# stages[5].trunc.threshold = 1e-15
# stages[5].convMinEnergyDiff = 1e-08
# stages[5].convergenceMinSweeps = 15
# stages[5].mode.TwoSite

# #7th stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[6].trunc.maxStates = 1024
# stages[6].convergenceMaxSweeps = 50
# stages[6].trunc.weight = 1e-14
# stages[6].trunc.threshold = 1e-15
# stages[6].convMinEnergyDiff = 1e-08
# stages[6].convergenceMinSweeps = 10
# stages[6].mode.DMRG3S

# #8th stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[7].trunc.maxStates = 2048
# stages[7].convergenceMaxSweeps = 20
# stages[7].trunc.weight = 1e-15
# stages[7].trunc.threshold = 1e-15
# stages[7].convMinEnergyDiff = 1e-09
# stages[7].convergenceMinSweeps = 5
# stages[7].mode.DMRG3S

# #9th stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[8].trunc.maxStates = 4096
# stages[8].convergenceMaxSweeps = 20
# stages[8].trunc.weight = 1e-15
# stages[8].trunc.threshold = 1e-15
# stages[8].convMinEnergyDiff = 1e-09
# #stages[8].convergenceMinSweeps = 5
# stages[8].mode.DMRG3S

# # assign stages to DMRG configuration object
# conf.stages = stages
# dmrg= ptn.mp.dmrg.PDMRG(vac_state.copy(), [h_tot], conf)

# # iterate over stages in config object
# energy_during_dmrg = []
# for m in conf.stages:
#     # run stage until either convergence is met or max. number of sweeps
#     vac_state = dmrg.run()

# for site in range(8):
#     bdim = vac_state[site].getTotalDims()[1]
#     print( 'bdim on site {} is {}'.format(site, bdim) )    
########### END GS CALCULATION

#Lindblad operators
def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

def lindblad_op_list_left_lead( Om_kl, delta_l, mu_l, T_l ):
    l_list_left = []
    l_list_left.append( np.sqrt( delta_l * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l ) ) * lat.get( 'ch',1 ) * lat.get( 'c',0 ) )
    l_list_left.append( np.sqrt( delta_l * fermi_dist( 1./T_l, Om_kl, mu_l)) * lat.get('c',1) * lat.get('ch',0) )
    return l_list_left

def lindblad_op_list_right_lead( Om_kr, delta_r, mu_r, T_r ):
    l_list_right = []
    l_list_right.append( np.sqrt( delta_r * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) ) * lat.get( 'ch',7 ) * lat.get( 'c',6 ) )
    l_list_right.append( np.sqrt( delta_r * fermi_dist( 1./T_r, Om_kr, mu_r)) * lat.get('c',7) * lat.get('ch',6) )
    return l_list_right

l_list_left = lindblad_op_list_left_lead( Om_kl, delta_l, mu_l, T_l )
l_list_right = lindblad_op_list_right_lead( Om_kr, delta_r, mu_r, T_r )
l_list = l_list_left + l_list_right

#Observables
obsdict = observables.ObservablesDict()
obsdict.initialize_observable('n',(8,), n_timesteps) 
obsdict.initialize_observable('block_entropies',(7,), n_timesteps)
obsdict.initialize_observable('rdm_phon',(max_bosons + 1, max_bosons + 1), n_timesteps) 
obsdict.initialize_observable('bond_dim',(8,), n_timesteps)
obsdict.initialize_observable('phonon_entanglement_entropy',(1,), n_timesteps) 
obsdict.initialize_observable('phonon_energy',(1,), n_timesteps) 
obsdict.initialize_observable('dot_energy',(1,), n_timesteps) 
obsdict.initialize_observable('phys_dim',(8,), n_timesteps) 

def compute_n(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    for site in range(8):
        obs_array[site] = np.real( ptn.mp.expectation(state, lat.get('n', site) ) ) #/ state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

def compute_block_entropies(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = state.block_entropies()
    #OBS DEPENDENT PART END
    return obs_array

def compute_rdm_phon(state, obs_array_shape,dtype = 'complex'):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    rdm = np.array(ptn.mp.rdm.o1rdm(state,4) )
    obs_array[ :rdm.shape[0], :rdm.shape[1] ] = rdm 
    #OBS DEPENDENT PART END
    return obs_array

def compute_bond_dim(state, obs_array_shape,dtype):  
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART STAR
    for site in range(len(obs_array)):
        obs_array[site] = state[site].getTotalDims()[2]
    #OBS DEPENDENT PART END
    return obs_array

def compute_phonon_entanglement_entropy(state, obs_array_shape,dtype = 'complex'):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    rdm = np.array( ptn.mp.rdm.o1rdm( state, 4) )
    R = rdm * ( sla.logm( rdm )/ sla.logm( np.matrix( [ [ 2 ] ] ) ) )
    S = - np.matrix.trace(R)
    obs_array = S
    #OBS DEPENDENT PART END
    return obs_array

def compute_phonon_energy(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = np.real( ptn.mp.expectation(state, h_boson ) ) #/ state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

def compute_dot_energy(state, obs_array_shape,dtype):  #EXAMPLE 1D
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART START
    obs_array = np.real( ptn.mp.expectation(state, h_s ) ) #/ state.norm() ** 2 #NOTE: state is in general not normalized
    #OBS DEPENDENT PART END
    return obs_array

def compute_phys_dim(state, obs_array_shape,dtype):  
    obs_array = np.zeros( obs_array_shape, dtype=dtype)
    #OBS DEPENDENT PART STAR
    for site in range(len(obs_array)):
        obs_array[site] = state[site].getTotalDims()[0]
    #OBS DEPENDENT PART END
    return obs_array

obsdict.add_observable_computing_function('n', compute_n )
obsdict.add_observable_computing_function('block_entropies', compute_block_entropies )
obsdict.add_observable_computing_function('rdm_phon', compute_rdm_phon )
obsdict.add_observable_computing_function('bond_dim', compute_bond_dim )
obsdict.add_observable_computing_function('phonon_entanglement_entropy', compute_phonon_entanglement_entropy )
obsdict.add_observable_computing_function('phonon_energy', compute_phonon_energy )
obsdict.add_observable_computing_function('dot_energy', compute_dot_energy )
obsdict.add_observable_computing_function('phys_dim', compute_phys_dim )

########TDVP CONFIG
conf_tdvp = ptn.tdvp.Conf()
conf_tdvp.mode = ptn.tdvp.Mode.GSE   #TwoSite, GSE, Subspace
conf_tdvp.dt = dt
conf_tdvp.trunc.threshold = 1e-10  #NOTE: set to zero for gse
conf_tdvp.trunc.weight = 1e-15 #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp.trunc.maxStates = 1000
conf_tdvp.exp_conf.errTolerance = 1e-10
conf_tdvp.exp_conf.inxTolerance = 1e-15
conf_tdvp.exp_conf.maxIter =  10
conf_tdvp.cache = 1
conf_tdvp.maxt = t_max

conf_tdvp.gse_conf.mode = ptn.tdvp.GSEMode.BeforeTDVP
conf_tdvp.gse_conf.krylov_order = 3 #FIXME 3,5 INCRESE
conf_tdvp.gse_conf.trunc_op = ptn.Truncation(1e-8 , maxStates=500) #maxStates shuld be the same as the one used for tdvp! 1e-8 - 1e-6
conf_tdvp.gse_conf.trunc_expansion = ptn.Truncation(1e-6, maxStates=500) #precision of GSE. par is trunc. treshold. do not goe below 10^-12 (numerical instability)!!
conf_tdvp.gse_conf.adaptive = True
conf_tdvp.gse_conf.sing_val_thresholds = [1e-12] # [1e-12] #most highly model-dependet parameter 


#compute time-evolution for one trajectory

qj = mps_quantum_jumps.MPSQuantumJumps( 8, lat, h_tot, [] ) # l_list, [ lat.get('c',1) * lat.get('ch',0),  lat.get('ch',1) * lat.get('c',0) ]

os.chdir('data_qj_mps')
first_trajectory = first_trajectory  #+ rank  NOTE: uncomment "+ rank" when parallelizing

######
energy_bos_t0 = ptn.mp.expectation(vac_state, h_boson)
nb_t0 = ptn.mp.expectation(vac_state, lat.get('nb',4)) 
ah_a_t0 = ptn.mp.expectation(vac_state, lat.get('a',4) * lat.get('ah',4) ) 
ah_a_ah_a_t0 = ptn.mp.expectation(vac_state, lat.get('ah',4) * lat.get('a',4) * lat.get('ah',5) * lat.get('a',5) ) 


#COMPUTE ONE TRAJECTORY WITH TDVP 
for trajectory in range(first_trajectory, first_trajectory + n_trajectories): 
    print('computing trajectory {}'.format(trajectory))
    test_singlet_traj_evolution = qj.quantum_jump_single_trajectory_time_evolution(vac_state, conf_tdvp, t_max, dt, trajectory, obsdict)

read_directory = os.getcwd()
write_directory = os.getcwd()

obsdict.compute_trajectories_averages_and_errors( list( range( first_trajectory, first_trajectory + n_trajectories) ), os.getcwd(), os.getcwd(), remove_single_trajectories_results=True ) 


#PLOT
# n_av = np.load('n_av.npy')

# fig, ax = plt.subplots()
# ax.plot(time_v, n_av[2,:-1], label='n_sys')

# plt.legend()
# fig.savefig('test_mps.png')
#########################