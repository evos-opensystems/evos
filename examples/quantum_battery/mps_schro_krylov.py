"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with mps quantum jumps. The dimension of the oscillator needs to be strongly truncated.
"""
import evos.src.methods.mps_schroedinger as mps_schroedinger
import evos.src.observables.observables_pickled as observables
import pyten as ptn
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
from scipy import linalg as sla
import argparse

sys.stdout.write('test')

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

#Lattice
ferm_bos_sites = [ 1, 1, 1, 1, 0, 1, 1 ] #doubled the fermionic sites to project-purify
lat = ptn.mp.lat.u1u1.genSpinlessFermiBose(ferm_bos_sites, max_bosons)
lat = ptn.mp.proj_pur.proj_purification(lat, [0], ["a", "ah"])
# print(lat)
vac_state =  ptn.mp.proj_pur.generateNearVacuumState(lat, 2, "0," + str( max_bosons ) )

#prepare PP vacuum for fermions
vac_state *= lat.get('ch',1)    
vac_state *= lat.get('ch',3) 
vac_state *= lat.get('ch',7)    

#FIXME: exite one particle in the left lead (USED TO DEBUGG WITHOUT INJECTION)
vac_state *= lat.get('ch',0) * lat.get('c',1) 
###############
# for site in range(8):
#     print(site, ptn.mp.expectation(vac_state, lat.get('n',site)))
# quit()    
###############       
#Hamiltonian
class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
    
    def h_s(self, eps): #system
        h_s = eps * lat.get('nf',2) 
        return h_s 
        
    def h_b(self, Om_kl, Om_kr): #leads
        h_b = Om_kl * lat.get('nf',0) + Om_kr * lat.get('nf',6)
        return h_b    
    
    def h_t(self, g_kl, g_kr): #system-leads
        # h_t = g_kl * (lat.get('ch',3) * lat.get('c',2) * lat.get('c',1) * lat.get('ch',0) + lat.get('ch',1) * lat.get('c',0) * lat.get('c',3) * lat.get('ch',2) )
        # h_t += g_kr * ( lat.get('ch',7) * lat.get('c',6) * lat.get('c',3) *lat.get('ch',2) + lat.get('ch',3) * lat.get('c',2) * lat.get('c',7) * lat.get('ch',6) )
        
        h_t = g_kl * ( lat.get('c',2) * lat.get('ch',0) +  lat.get('c',0) * lat.get('ch',2) )
        h_t += g_kr * (  lat.get('c',6) * lat.get('ch',2) + lat.get('c',2) * lat.get('ch',6) )
        
        return h_t
    
    def h_boson(self, om_0): #oscillator
        h_boson = om_0 * lat.get('nb',4) 
        return h_boson
    
    def h_v(self, F, N0): #system-oscillator
        h_v = - F * ( lat.get('nf',2) - N0 * lat.get('I') ) * ( lat.get('ah',4) * lat.get('a',5)  + lat.get('a',4) * lat.get('ah',5) )
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F):
        h_tot =  self.h_boson(om_0) + self.h_v(F, N0) + self.h_s(eps) + self.h_t(g_kl, g_kr) + self.h_b(Om_kl, Om_kr) 
        #h_tot.truncate()
        return h_tot
    
#Build Hamiltonian
ham = Hamiltonian(lat, max_bosons)
h_s = ham.h_s(eps)
# h_b = ham.h_b(Om_kl, Om_kr)
# h_t = ham.h_t(g_kl, g_kr)
h_boson = ham.h_boson(om_0)
# h_v = ham.h_v(F, N0)
# h_tot = h_s + h_b + h_t + h_boson + h_v

h_tot = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F)

# lat.add('h_tot', 'h_tot', h_tot)
# lat.save('lat')
# quit()

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

########KRYLOV CONFIG
conf_krylov = ptn.krylov.Conf()
conf_krylov.dt = dt
conf_krylov.threshold= 1e-8
conf_krylov.weight= 1e-10
conf_krylov.errTolerance= 1e-7
conf_krylov.inxTolerance= 1e-6
conf_krylov.maxIter = 10
conf_krylov.maxStates = 1000
conf_krylov.tend = t_max

os.chdir('data_schro_kry_mps')

#COMPUTE TEVO WITH KRYLOV AND EVOS SCHROEDINGER METHOD
mps_schroedinger.MPSSchroedinger( 8, lat, h_tot ).schroedinger_time_evolution( vac_state, obsdict, [], krylov = True, krylov_config = conf_krylov, save_states = True )
os.system('rm *.mps')