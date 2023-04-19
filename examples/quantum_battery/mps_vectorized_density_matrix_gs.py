"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with mps vectorized density matrix.
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
N0 = 0 #FIXME: is this correct?
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
ferm_bos_sites = [ 1, 1, 1, 1, 0, 1, 1,   1, 1, 1, 1, 0, 1, 1 ] #doubled the fermionic sites to project-purify by hand 
lat = ptn.mp.lat.u1u1.genSpinlessFermiBose(ferm_bos_sites, max_bosons)
lat = ptn.mp.proj_pur.proj_purification(lat, [0], ["a", "ah"])

vac_state =  ptn.mp.proj_pur.generateNearVacuumState(lat, 2, "0," + str( max_bosons ) )

#prepare PP vacuum for fermions
idx_shift_lattice_doubling = 8 
vac_state *= lat.get('ch',1)    
vac_state *= lat.get('ch',3) 
vac_state *= lat.get('ch',7)    
vac_state *= lat.get('ch',1 + idx_shift_lattice_doubling)    
vac_state *= lat.get('ch',3 + idx_shift_lattice_doubling) 
vac_state *= lat.get('ch',7 + idx_shift_lattice_doubling)  

#FIXME: I AM NOT EXCITING ANY PARTICLE HERE. CHANGE THAT IF YOU WANT TO TEST THE HAMILTONIAN ONLY!

class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps, idx_shift_lattice_doubling): #system
        h_s = eps * lat.get('nf',2 + idx_shift_lattice_doubling)  #no need to PP
        #h_s = ptn.mp.addLog(h_s)
        #h_s.truncate()
        return h_s 

    def h_b(self, Om_kl, Om_kr, idx_shift_lattice_doubling): #leads
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = []
        h_b.append( Om_kl * lat.get('nf',0 + idx_shift_lattice_doubling) ) #no need to PP
        h_b.append( Om_kr * lat.get('nf',6 + idx_shift_lattice_doubling) ) #no need to PP
        h_b = ptn.mp.addLog(h_b)
        #h_b.truncate()
        return h_b
   
    def h_t(self, g_kl, g_kr, idx_shift_lattice_doubling): #system-leads
        
        h_t = g_kl * ( lat.get('ch',1 + idx_shift_lattice_doubling) * lat.get('c',0 + idx_shift_lattice_doubling) * lat.get('c',3 + idx_shift_lattice_doubling) * lat.get('ch',2 + idx_shift_lattice_doubling)  +  lat.get('ch',3 + idx_shift_lattice_doubling) * lat.get('c',2 + idx_shift_lattice_doubling) * lat.get('c',1 + idx_shift_lattice_doubling) * lat.get('ch',0 + idx_shift_lattice_doubling) ) 
        h_t += g_kr * ( lat.get('ch',7 + idx_shift_lattice_doubling) * lat.get('c',6 + idx_shift_lattice_doubling)  * lat.get('c',3 + idx_shift_lattice_doubling) * lat.get('ch',2 + idx_shift_lattice_doubling)  + lat.get('ch',3 + idx_shift_lattice_doubling) * lat.get('c',2 + idx_shift_lattice_doubling) * lat.get('c',7 + idx_shift_lattice_doubling) * lat.get('ch',6 + idx_shift_lattice_doubling) ) 
    
        #h_t.truncate()
        return h_t
    
    def h_boson(self, om_0, idx_shift_lattice_doubling): #oscillator
        h_boson = om_0 * lat.get('nb',4 + idx_shift_lattice_doubling) 
        return h_boson
    
    def h_v(self, F, N0, idx_shift_lattice_doubling): #system-oscillator
        h_v = - F * ( lat.get('nf',2 + idx_shift_lattice_doubling) - N0 * lat.get('I') ) *  ( lat.get('ah',4 + idx_shift_lattice_doubling) * lat.get('a',5 + idx_shift_lattice_doubling)  + lat.get('a',4 + idx_shift_lattice_doubling) * lat.get('ah',5 + idx_shift_lattice_doubling) ) 
        #h_v.truncate()
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, idx_shift_lattice_doubling=0):
        h_tot =  self.h_boson(om_0, idx_shift_lattice_doubling) + self.h_v(F, N0, idx_shift_lattice_doubling) + self.h_s(eps, idx_shift_lattice_doubling) + self.h_t(g_kl, g_kr, idx_shift_lattice_doubling) + self.h_b(Om_kl, Om_kr, idx_shift_lattice_doubling) 
        #h_tot.truncate()
        return h_tot

#Hamiltonian
ham = Hamiltonian(lat, max_bosons)        
h_tot_left = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, idx_shift_lattice_doubling = 0)
h_tot_right = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, idx_shift_lattice_doubling = 8)


# for site in range(16):
#     print('bdim on site {} = {}'.format( site, (h_tot_left + h_tot_right )[site].getTotalDims()[0]) )
# quit()

#BUILD UNNORMALIZED PURIFIED IDENTITY
def mpo_max_ent_pair_ferm(site):
    """_summary_
    """
    
    op = lat.get('I')
    op_tot = op.copy()
    for mode in range(1,2):
        print('mode = ', mode)
        op *= lat.get('c',site + 1 ) * lat.get('ch',site)  *  lat.get('c',site + idx_shift_lattice_doubling + 1) * lat.get('ch',site + idx_shift_lattice_doubling) #* 1./mode #FIXME: reverse order?
        op *= 1./mode
        op_tot += op
        op.truncate()
        op_tot.truncate()
        
    return op_tot  

def mpo_max_ent_pair_bos(site, max_bosons):
    """_summary_
    """
    op = lat.get('I')
    op_tot = op.copy()
    for mode in range(1, max_bosons+1):
        print('mode = ', mode)
        op *= lat.get('a',site + 1 ) * lat.get('ah',site)  *  lat.get('a',site + idx_shift_lattice_doubling + 1) * lat.get('ah',site + idx_shift_lattice_doubling) #* 1./mode #FIXME: reverse order?
        op *= 1./mode
        op_tot += op
        op.truncate()
        op_tot.truncate()
        
    return op_tot   


purified_id = vac_state.copy() 

for site in np.arange(0, 8, 2):
    print('on site ', site)
    if site != 4:
        print( 'applying "mpo_max_ent_pair_ferm" on site {}'.format(site) )
        purified_id *=  mpo_max_ent_pair_ferm(site)
    
    elif site == 4:
        print( 'applying "mpo_max_ent_pair_bos" on site {}'.format(site) )    
        purified_id *= mpo_max_ent_pair_bos(site, max_bosons)
        

###purified_id.normalise() #FIXME
# for site in range(16):
#     print(site, ptn.mp.expectation(purified_id, lat.get('n',site)))
# quit()           

#VECTORIZED DISSIPATOR
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


def compute_vectorized_dissipator():
    
    first_term =  delta_l * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l )  * lat.get( 'ch',1 ) * lat.get( 'c',0 )   *    lat.get( 'ch',1 + idx_shift_lattice_doubling ) * lat.get( 'c',0 + idx_shift_lattice_doubling )
    
    first_term += delta_l * fermi_dist( 1./T_l, Om_kl, mu_l) * lat.get('c',1) * lat.get('ch',0)  *   lat.get('c',1+ idx_shift_lattice_doubling) * lat.get('ch',0+ idx_shift_lattice_doubling)

    first_term += delta_r * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) * lat.get( 'ch',7 ) * lat.get( 'c',6 )   *  lat.get( 'ch',7+ idx_shift_lattice_doubling ) * lat.get( 'c',6+ idx_shift_lattice_doubling ) 
    
    first_term +=  delta_r * fermi_dist( 1./T_r, Om_kr, mu_r) * lat.get('c',7) * lat.get('ch',6) * lat.get('c',7 + idx_shift_lattice_doubling) * lat.get('ch',6+ idx_shift_lattice_doubling)
    
    
    second_term = delta_l * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l )  * lat.get( 'ch',1 ) * lat.get( 'c',0 )   *    lat.get( 'c',1  ) * lat.get( 'ch',0 )
    
    second_term += delta_l * fermi_dist( 1./T_l, Om_kl, mu_l) * lat.get('c',1) * lat.get('ch',0)  *   lat.get('ch',1) * lat.get('c',0)
    
    second_term += delta_r * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) * lat.get( 'ch',7 ) * lat.get( 'c',6 )   *  lat.get( 'c',7 ) * lat.get( 'ch',6 ) 
    
    second_term += delta_r * fermi_dist( 1./T_r, Om_kr, mu_r) * lat.get('c',7) * lat.get('ch',6) * lat.get('ch',7) * lat.get('c',6)
    second_term *= 0.5
    
    
    third_term = delta_l * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l )  * lat.get( 'ch',1+ idx_shift_lattice_doubling ) * lat.get( 'c',0+ idx_shift_lattice_doubling )   *    lat.get( 'c',1+ idx_shift_lattice_doubling  ) * lat.get( 'ch',0 + idx_shift_lattice_doubling)
    
    third_term += delta_l * fermi_dist( 1./T_l, Om_kl, mu_l) * lat.get('c',1+ idx_shift_lattice_doubling) * lat.get('ch',0+ idx_shift_lattice_doubling)  *   lat.get('ch',1+ idx_shift_lattice_doubling) * lat.get('c',0+ idx_shift_lattice_doubling)
    
    third_term += delta_r * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) * lat.get( 'ch',7+ idx_shift_lattice_doubling ) * lat.get( 'c',6+ idx_shift_lattice_doubling )   *  lat.get( 'c',7 + idx_shift_lattice_doubling) * lat.get( 'ch',6+ idx_shift_lattice_doubling ) 
    
    third_term += delta_r * fermi_dist( 1./T_r, Om_kr, mu_r) * lat.get('c',7+ idx_shift_lattice_doubling) * lat.get('ch',6+ idx_shift_lattice_doubling) * lat.get('ch',7+ idx_shift_lattice_doubling) * lat.get('c',6+ idx_shift_lattice_doubling)
    
    third_term *= 0.5
    
    vectorized_dissipator = first_term + second_term + third_term
    vectorized_dissipator.truncate()
    return vectorized_dissipator


vectorized_dissipator = compute_vectorized_dissipator()    
vectorized_lindbladian = -1j*h_tot_left +1j*h_tot_right + vectorized_dissipator 
vectorized_lindbladian.truncate()
vectorized_lindbladian_dag = ptn.mp.dot( lat.get("I"), vectorized_lindbladian.copy() )
vectorized_L_dag_L = vectorized_lindbladian * vectorized_lindbladian_dag
vectorized_L_dag_L.truncate()
lat.add('vectorized_L_dag_L', 'vectorized_L_dag_L', vectorized_L_dag_L)
lat.save('lat')

# for site in range(16):
#     print('bdim on site {} = {}'.format( site, vectorized_L_dag_L[site].getTotalDims()[0]) )

#GROUND STATE OF HAMILTONIAN AS WARM UP FOR GS FOR L_DAGGER_L !
init_state_dmrg_h = vac_state.copy()
init_state_dmrg_h *=  lat.get('c',1) * lat.get('ch',0)
init_state_dmrg_h *= lat.get('c',3) * lat.get('ch',2)
init_state_dmrg_h *=  lat.get('c',9) * lat.get('ch',8)
init_state_dmrg_h *= lat.get('c',11) * lat.get('ch',10)

init_state_dmrg_h *=  lat.get('a',5) * lat.get('ah',4)
init_state_dmrg_h *=  lat.get('a',5) * lat.get('ah',4)
init_state_dmrg_h *=  lat.get('a',13) * lat.get('ah',12)
init_state_dmrg_h *=  lat.get('a',13) * lat.get('ah',12)

init_state_dmrg_h.normalise()
# print( 'init_state_dmrg_h.norm() = ',init_state_dmrg_h.norm() )
# quit()
conf = ptn.dmrg.DMRGConfig()
# give us a list to add stages
stages = []

#first stage
stages.append(ptn.dmrg.DMRGStage())
stages[0].trunc.maxStates = 16
stages[0].convergenceMaxSweeps = 200
stages[0].trunc.weight = 1e-6
stages[0].trunc.threshold = 1e-8
stages[0].convergenceMinSweeps = 50
#stages[0].convMinEnergyDiff = -1
stages[0].mode.DMRG3S
#second stage
stages.append(ptn.dmrg.DMRGStage())
stages[1].trunc.maxStates = 32
stages[1].convergenceMaxSweeps = 150
stages[1].trunc.weight = 1e-7
stages[1].trunc.threshold = 1e-9
stages[1].convergenceMinSweeps = 40
#stages[1].convMinEnergyDiff = -1
stages[1].mode.DMRG3S

#third stage
stages.append(ptn.dmrg.DMRGStage())
stages[2].trunc.maxStates = 64
stages[2].convergenceMaxSweeps = 100
stages[2].trunc.weight = 1e-8
stages[2].trunc.threshold = 1e-10
stages[2].convergenceMinSweeps = 30
#[2].convMinEnergyDiff = -1
stages[2].mode.TwoSite

#fourth stage
stages.append(ptn.dmrg.DMRGStage())
stages[3].trunc.maxStates = 128
stages[3].convergenceMaxSweeps = 100
stages[3].trunc.weight = 1e-10
stages[3].trunc.threshold = 1e-12
stages[3].convergenceMinSweeps = 25
#stages[3].convMinEnergyDiff = -1
stages[3].mode.DMRG3S

#fifth stage
stages.append(ptn.dmrg.DMRGStage())
stages[4].trunc.maxStates = 256
stages[4].convergenceMaxSweeps = 100
stages[4].trunc.weight = 1e-11
stages[4].trunc.threshold = 1e-13
stages[4].convergenceMinSweeps = 20
#stages[4].convMinEnergyDiff = -1
stages[4].mode.DMRG3S

#6th stage
stages.append(ptn.dmrg.DMRGStage())
stages[5].trunc.maxStates = 512
stages[5].convergenceMaxSweeps = 100
stages[5].trunc.weight = 1e-13
stages[5].trunc.threshold = 1e-15
stages[5].convMinEnergyDiff = 1e-08
stages[5].convergenceMinSweeps = 15
stages[5].mode.TwoSite

#7th stage
stages.append(ptn.dmrg.DMRGStage())
stages[6].trunc.maxStates = 1024
stages[6].convergenceMaxSweeps = 50
stages[6].trunc.weight = 1e-14
stages[6].trunc.threshold = 1e-15
stages[6].convMinEnergyDiff = 1e-08
stages[6].convergenceMinSweeps = 10
stages[6].mode.DMRG3S

#8th stage
stages.append(ptn.dmrg.DMRGStage())
stages[7].trunc.maxStates = 2048
stages[7].convergenceMaxSweeps = 20
stages[7].trunc.weight = 1e-15
stages[7].trunc.threshold = 1e-15
stages[7].convMinEnergyDiff = 1e-09
stages[7].convergenceMinSweeps = 5
stages[7].mode.DMRG3S

#9th stage
stages.append(ptn.dmrg.DMRGStage())
stages[8].trunc.maxStates = 4096
stages[8].convergenceMaxSweeps = 20
stages[8].trunc.weight = 1e-15
stages[8].trunc.threshold = 1e-15
stages[8].convMinEnergyDiff = 1e-09
#stages[8].convergenceMinSweeps = 5
stages[8].mode.DMRG3S

# assign stages to DMRG configuration object
conf.stages = stages
dmrg= ptn.mp.dmrg.PDMRG(init_state_dmrg_h.copy(), [h_tot_left + h_tot_right], conf)

# iterate over stages in config object
energy_during_dmrg = []
for m in conf.stages:
    # run stage until either convergence is met or max. number of sweeps
    h_gs = dmrg.run()

########### END GS CALCULATION




#GROUND STATE OF  L_DAGGER_L STARTING FROM GS OF HAMILTONIAN

conf = ptn.dmrg.DMRGConfig()
# give us a list to add stages
stages = []

#first stage
stages.append(ptn.dmrg.DMRGStage())
stages[0].trunc.maxStates = 16
stages[0].convergenceMaxSweeps = 200
stages[0].trunc.weight = 1e-6
stages[0].trunc.threshold = 1e-8
stages[0].convergenceMinSweeps = 50
#stages[0].convMinEnergyDiff = -1
stages[0].mode.DMRG3S
#second stage
stages.append(ptn.dmrg.DMRGStage())
stages[1].trunc.maxStates = 32
stages[1].convergenceMaxSweeps = 150
stages[1].trunc.weight = 1e-7
stages[1].trunc.threshold = 1e-9
stages[1].convergenceMinSweeps = 40
#stages[1].convMinEnergyDiff = -1
stages[1].mode.DMRG3S

#third stage
stages.append(ptn.dmrg.DMRGStage())
stages[2].trunc.maxStates = 64
stages[2].convergenceMaxSweeps = 100
stages[2].trunc.weight = 1e-8
stages[2].trunc.threshold = 1e-10
stages[2].convergenceMinSweeps = 30
#[2].convMinEnergyDiff = -1
stages[2].mode.TwoSite

#fourth stage
stages.append(ptn.dmrg.DMRGStage())
stages[3].trunc.maxStates = 128
stages[3].convergenceMaxSweeps = 100
stages[3].trunc.weight = 1e-10
stages[3].trunc.threshold = 1e-12
stages[3].convergenceMinSweeps = 25
#stages[3].convMinEnergyDiff = -1
stages[3].mode.DMRG3S

#fifth stage
stages.append(ptn.dmrg.DMRGStage())
stages[4].trunc.maxStates = 256
stages[4].convergenceMaxSweeps = 100
stages[4].trunc.weight = 1e-11
stages[4].trunc.threshold = 1e-13
stages[4].convergenceMinSweeps = 20
#stages[4].convMinEnergyDiff = -1
stages[4].mode.DMRG3S

#6th stage
stages.append(ptn.dmrg.DMRGStage())
stages[5].trunc.maxStates = 512
stages[5].convergenceMaxSweeps = 100
stages[5].trunc.weight = 1e-13
stages[5].trunc.threshold = 1e-15
stages[5].convMinEnergyDiff = 1e-08
stages[5].convergenceMinSweeps = 15
stages[5].mode.TwoSite

#7th stage
stages.append(ptn.dmrg.DMRGStage())
stages[6].trunc.maxStates = 1024
stages[6].convergenceMaxSweeps = 50
stages[6].trunc.weight = 1e-14
stages[6].trunc.threshold = 1e-15
stages[6].convMinEnergyDiff = 1e-08
stages[6].convergenceMinSweeps = 10
stages[6].mode.DMRG3S

#8th stage
stages.append(ptn.dmrg.DMRGStage())
stages[7].trunc.maxStates = 2048
stages[7].convergenceMaxSweeps = 20
stages[7].trunc.weight = 1e-15
stages[7].trunc.threshold = 1e-15
stages[7].convMinEnergyDiff = 1e-09
stages[7].convergenceMinSweeps = 5
stages[7].mode.DMRG3S

#9th stage
stages.append(ptn.dmrg.DMRGStage())
stages[8].trunc.maxStates = 4096
stages[8].convergenceMaxSweeps = 20
stages[8].trunc.weight = 1e-15
stages[8].trunc.threshold = 1e-15
stages[8].convMinEnergyDiff = 1e-09
#stages[8].convergenceMinSweeps = 5
stages[8].mode.DMRG3S


#10th stage
stages.append(ptn.dmrg.DMRGStage())
stages[9].trunc.maxStates = 8192
stages[9].convergenceMaxSweeps = 20
stages[9].trunc.weight = 1e-15
stages[9].trunc.threshold = 1e-15
stages[9].convMinEnergyDiff = 1e-09
stages[9].mode.DMRG3S

# assign stages to DMRG configuration object
conf.stages = stages
dmrg= ptn.mp.dmrg.PDMRG(h_gs.copy(), [vectorized_L_dag_L], conf)

# iterate over stages in config object
energy_during_dmrg = []
for m in conf.stages:
    # run stage until either convergence is met or max. number of sweeps
    ness_mps = dmrg.run()

########### END GS CALCULATION