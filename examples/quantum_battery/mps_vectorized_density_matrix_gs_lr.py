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
import argparse

arg_parser = argparse.ArgumentParser(description = "Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with ed lindblad. The dimension of the oscillator needs to be strongly truncated.")
arg_parser.add_argument("-b",   "--bosons", dest = 'max_bosons',  default = 4, type = int, help = 'number of bosonic degrees of freedom - 1 [4]')
arg_parser.add_argument("-mu_l",   "--checmical_pot_left_lead", dest = 'mu_l',  default = +0.5, type = float, help = 'checmical pot. left lead [0.5]')
arg_parser.add_argument("-mu_r",   "--checmical_pot_right_lead", dest = 'mu_r',  default = -0.5, type = float, help = 'checmical pot. right lead [-0.5]')

#FIXME: ADD MU_L AND MU_R
args = arg_parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)
#sys.stdout.write('test')

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
N0 = 0.5 #0.5FIXME: is this correct?
delta_l = 1
delta_r = 1

mu_l = args.mu_l
mu_r = args.mu_r

T_l = 1./0.5 #FIXME: beta_l = 0.5
T_r = 1./0.5 #FIXME: beta_r = 0.5
k_b = 1 #boltzmann constant

even_odd_idx_shift = 1

################
def make_writing_dir_and_change_to_it( parent_data_dirname: str, parameter_dict: dict, overwrite: bool = False, create_directory: bool = True ) -> str :
    """given a dictionary with some selected job's parameters, it creates the correct subfolder in which to run the job and changes to it

    Parameters
    ----------
    parent_data_dirname : str
        name of the parent directory
    parameter_dict : dict
        parameter dictionary specifying the directory

    Returns
    -------
    str
        path of the directory in which to write the states or the observables
    """
    import os 
    from datetime import date

    #go to parent folder if existing. create one with date attached and go to it if not existing
    if os.path.isdir(parent_data_dirname):
            os.chdir(parent_data_dirname)
    else:
        if create_directory:
            parent_data_dirname += '_'+str( date.today() )
            os.mkdir( parent_data_dirname )
            os.chdir(parent_data_dirname)


    dir_depth = len(parameter_dict)
    count_dir_depth = 0
    for par in parameter_dict:
        subdir_name = par +'_'+str(parameter_dict[par])

        #if reached lowest directory level AND it already exists
        if count_dir_depth == dir_depth-1 and os.path.isdir(subdir_name): 
            #print(subdir_name)
            if not overwrite and create_directory: 
                subdir_name += '_'+str( date.today() )
        
        #all other directory levels OR the lowest but it doesn't exists
        if os.path.isdir(subdir_name):
            os.chdir(subdir_name)
        else:
            if create_directory:
                os.mkdir( subdir_name )
                os.chdir(subdir_name)
                
        count_dir_depth += 1

    writing_dir = os.getcwd()
    return writing_dir

parameter_dict = {'max_bosons': max_bosons,'mu_l' : mu_l, 'mu_r' : mu_r  }
writing_dir = make_writing_dir_and_change_to_it('data_mps_lindblad_ss', parameter_dict, overwrite=True)

################


#Lattice
ferm_bos_sites = [0,0,0,0,1,1,0,0]
lat = ptn.mp.lat.u1.genSpinlessFermiBose_NilxU1( ferm_bos_sites, max_bosons)
lat = ptn.mp.proj_pur.proj_purification(lat, [1], ["a", "ah"])

#Vacuum state
vac_state =  ptn.mp.proj_pur.generateNearVacuumState(lat, 2, str( max_bosons ) )

#destroy particles on fermionic sites
for site in [0,1,2,3,8,9]:
    vac_state *= lat.get('c',site)
    vac_state.normalise()
    
#for site in range(10):
    #print('<n> on site {} is {}'.format(site, ptn.mp.expectation(vac_state, lat.get('n',site) ) ) )
    #print('<n_b> on site {} is {}'.format(site, ptn.mp.expectation(vac_state, lat.get('nb',site) ) ) )
    

#Hamiltonian
class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps, even_odd_idx_shift): #system
        h_s = eps * lat.get('n',2 + even_odd_idx_shift)  
        #print('h_s on site ', 2 + even_odd_idx_shift)
        h_s.truncate()
        return h_s 

    def h_b(self, Om_kl, Om_kr, even_odd_idx_shift): #leads
        #NOTE: added mu_l and mu_rto onsite energies
        h_b =  Om_kl * lat.get('n',0 + even_odd_idx_shift) +  Om_kr * lat.get('n',8 + even_odd_idx_shift)
        #print('h_b on sites {} and {} '.format(even_odd_idx_shift, 8+even_odd_idx_shift))
        h_b.truncate()
        return h_b
   
    def h_t(self, g_kl, g_kr, even_odd_idx_shift): #system-leads
        
        h_t = g_kl * ( lat.get('c',2 + even_odd_idx_shift) * lat.get('ch',0 + even_odd_idx_shift) + lat.get('c',0 + even_odd_idx_shift) * lat.get('ch',2 + even_odd_idx_shift) ) 
        h_t += g_kr * ( lat.get('c',2 + even_odd_idx_shift) * lat.get('ch',8 + even_odd_idx_shift) + lat.get('c',8 + even_odd_idx_shift) * lat.get('ch',2 + even_odd_idx_shift) ) 
        # print('left hopping between {} and {}'.format(even_odd_idx_shift, 2 + even_odd_idx_shift))
        # print('left hopping between {} and {}'.format(8 + even_odd_idx_shift, 2 + even_odd_idx_shift))
        h_t.truncate()
        return h_t
    
    def h_boson(self, om_0, even_odd_idx_shift): #oscillator
        h_boson = om_0 * lat.get('nb',4 + 2*even_odd_idx_shift) 
        #print('h_boson on site {}'.format(4 + 2*even_odd_idx_shift))
        h_boson.truncate()
        return h_boson
    
    def h_v(self, F, N0, even_odd_idx_shift): #system-oscillator
        h_v = - F * ( lat.get('n',2 + even_odd_idx_shift) - N0 * lat.get('I') ) *  ( lat.get('ah',4 + 2*even_odd_idx_shift) * lat.get('a',5 + 2*even_odd_idx_shift)  + lat.get('a',4 + 2*even_odd_idx_shift) * lat.get('ah',5 + 2*even_odd_idx_shift) ) 
        h_v.truncate()
        #print('h_v between system-tls {} and physical {} and pp {} sites'. format(2 + even_odd_idx_shift,4 + 2*even_odd_idx_shift, 5 + 2*even_odd_idx_shift ))
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, even_odd_idx_shift=0):
        #print('even_odd_idx_shift = ', even_odd_idx_shift)
        h_tot =  self.h_s(eps, even_odd_idx_shift) + self.h_b(Om_kl, Om_kr, even_odd_idx_shift) + self.h_t(g_kl, g_kr, even_odd_idx_shift) + self.h_boson(om_0, even_odd_idx_shift) + self.h_v(F, N0, even_odd_idx_shift) 
        h_tot.truncate()
        return h_tot    
        
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)      
h_tot_even = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, even_odd_idx_shift = 0)
h_tot_even.truncate()
h_tot_odd = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, even_odd_idx_shift = even_odd_idx_shift)
h_tot_odd.truncate()

def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

def compute_vectorized_dissipator():
    
    vectorized_dissipator = delta_l * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l ) * ( lat.get('c',1) * lat.get('c',0) - 0.5 * lat.get('c',0) * lat.get('ch',0) - 0.5 * lat.get('c',1) * lat.get('ch',1) )#annihilator site 0
    vectorized_dissipator += delta_l * fermi_dist( 1./T_l, Om_kl, mu_l) * ( lat.get('ch',0) * lat.get('ch',1) - 0.5 * lat.get('ch',0) * lat.get('c',0) - 0.5 * lat.get('ch',1) * lat.get('c',1) ) #creator site 0

    vectorized_dissipator += delta_r * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) * ( lat.get('c',9) * lat.get('c',8) - 0.5 * lat.get('c',8) * lat.get('ch',8) - 0.5 * lat.get('c',9) * lat.get('ch',9) )#annihilator site 0
    vectorized_dissipator += delta_r * fermi_dist( 1./T_r, Om_kr, mu_r) * ( lat.get('ch',8) * lat.get('ch',9) - 0.5 * lat.get('ch',8) * lat.get('c',8) - 0.5 * lat.get('ch',9) * lat.get('c',9) )#creator site 0

    return vectorized_dissipator

vectorized_lindbladian = -1j*h_tot_even + 1j*h_tot_odd + compute_vectorized_dissipator() 
vectorized_lindbladian.truncate()
vectorized_lindbladian_dagger = ptn.mp.dot(lat.get("I"), vectorized_lindbladian.copy())
l_dagger_l = vectorized_lindbladian * vectorized_lindbladian_dagger
#l_dagger_l.truncate()

#BUILD UNNORMALIZED PURIFIED IDENTITY
def mpo_max_ent_pair_ferm(site):
    """_summary_
    """
    
    op = lat.get('I')
    op_tot = op.copy()
    for mode in range(1,2):
        #print('mode = ', mode)
        op *= lat.get('ch',site) * lat.get('ch',site + even_odd_idx_shift) 
        op *= 1./mode
        op_tot += op
        #op.truncate()
        #op_tot.truncate()
        
    return op_tot  

def mpo_max_ent_pair_bos(site, max_bosons):
    """_summary_
    """
    op = lat.get('I')
    op_tot = op.copy()
    for mode in range(1, max_bosons+1):
        #print('mode = ', mode)
        op *= lat.get('a',site + 1 ) * lat.get('ah',site)  *  lat.get('a',site + 2*even_odd_idx_shift + 1) * lat.get('ah',site + 2*even_odd_idx_shift) #* 1./mode #FIXME: reverse order?
        op *= 1./mode
        op_tot += op
        #op.truncate()
        #op_tot.truncate()
        
    return op_tot   

purified_id = vac_state.copy()     
for site in [0,2,8]:
    purified_id *=  mpo_max_ent_pair_ferm(site)
    purified_id.truncate()
for site in [4]:    
    purified_id *= mpo_max_ent_pair_bos(site, max_bosons)            
    #purified_id.truncate()


#GROUND STATE SEARCH PREPARATION: computation of fermi sea from tight-binding hamiltonian
# def compute_h_tight_binding():
#     """nearest-neighbour hopping between the 3 physical and between the 3 bath sites.
#     """
#     h_tight_binding = lat.get('c',2)*lat.get('ch',0) + lat.get('c',0)*lat.get('ch',2) + lat.get('c',8)*lat.get('ch',2) + lat.get('c',2)*lat.get('ch',8) #hopping between physical sites
#     h_tight_binding += lat.get('c',3)*lat.get('ch',1) + lat.get('c',1)*lat.get('ch',3) + lat.get('c',9)*lat.get('ch',3) + lat.get('c',3)*lat.get('ch',9) #hopping between bath sites
#     return h_tight_binding

# h_tight_binding = compute_h_tight_binding()


# #create state with two particles on physical sites and two on auxiliary sites
# #FIXME: what is the actual occupation for spinless fermi sea? 1/2?
# init_state_for_fermi_sea = vac_state.copy()
# init_state_for_fermi_sea *= lat.get('ch',0)
# init_state_for_fermi_sea.normalise()
# init_state_for_fermi_sea *= lat.get('ch',8)
# init_state_for_fermi_sea.normalise()
# init_state_for_fermi_sea *= lat.get('ch',1)
# init_state_for_fermi_sea.normalise()
# init_state_for_fermi_sea *= lat.get('ch',9)
# init_state_for_fermi_sea.normalise()

# #GS for Fermi sea
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

# #10th stage
# stages.append(ptn.dmrg.DMRGStage())
# stages[9].trunc.maxStates = 8192
# stages[9].convergenceMaxSweeps = 20
# stages[9].trunc.weight = 1e-15
# stages[9].trunc.threshold = 1e-15
# stages[9].convMinEnergyDiff = 1e-09
# stages[9].mode.DMRG3S

# # assign stages to DMRG configuration object
# conf.stages = stages
# dmrg= ptn.mp.dmrg.PDMRG(init_state_for_fermi_sea.copy(), [h_tight_binding], conf)  #vectorized_L_dag_L

# # iterate over stages in config object
# energy_during_dmrg = []
# for m in conf.stages:
#     # run stage until either convergence is met or max. number of sweeps
#     fermi_sea = dmrg.run()

########### END FERMI SEA CALCULATION


#GROUND STATE OF  L_DAGGER_L 
conf = ptn.dmrg.DMRGConfig()
# give us a list to add stages
stages = []

#first stage
stages.append(ptn.dmrg.DMRGStage())
stages[0].trunc.maxStates = 100
stages[0].convergenceMaxSweeps = 200
stages[0].trunc.weight = 1e-6
stages[0].trunc.threshold = 1e-8
stages[0].convergenceMinSweeps = 50
#stages[0].convMinEnergyDiff = -1
stages[0].mode.DMRG3S
#second stage
stages.append(ptn.dmrg.DMRGStage())
stages[1].trunc.maxStates = 150
stages[1].convergenceMaxSweeps = 150
stages[1].trunc.weight = 1e-7
stages[1].trunc.threshold = 1e-9
stages[1].convergenceMinSweeps = 40
#stages[1].convMinEnergyDiff = -1
stages[1].mode.DMRG3S

#third stage
stages.append(ptn.dmrg.DMRGStage())
stages[2].trunc.maxStates = 200
stages[2].convergenceMaxSweeps = 100
stages[2].trunc.weight = 1e-8
stages[2].trunc.threshold = 1e-10
stages[2].convergenceMinSweeps = 30
#[2].convMinEnergyDiff = -1
stages[2].mode.TwoSite

#fourth stage
stages.append(ptn.dmrg.DMRGStage())
stages[3].trunc.maxStates = 250
stages[3].convergenceMaxSweeps = 100
stages[3].trunc.weight = 1e-10
stages[3].trunc.threshold = 1e-12
stages[3].convergenceMinSweeps = 25
#stages[3].convMinEnergyDiff = -1
stages[3].mode.DMRG3S

#fifth stage
stages.append(ptn.dmrg.DMRGStage())
stages[4].trunc.maxStates = 300
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
dmrg= ptn.mp.dmrg.PDMRG(purified_id.copy(), [l_dagger_l], conf)  #fermi_sea, purified_id

# iterate over stages in config object
# print('purified_id.norm() = ',purified_id.norm() )
# quit()
norm_during_dmrg = []
for m in conf.stages:
    # run stage until either convergence is met or max. number of sweeps
    ness_mps = dmrg.run()
    norm_during_dmrg.append(ness_mps.norm())
    np.save('norm_during_dmrg',norm_during_dmrg)

########### END GS CALCULATION


#COMPUTE OBSERVABLES
#Compute trace-norm for observables
trace_norm_ness_mps = ptn.mp.overlap(purified_id, ness_mps)
    
n_exp = np.zeros(10)
n_b_exp = np.zeros( ( 10) )

for site in range(10):
    n_exp[site] = np.real( ptn.mp.expectation(purified_id, lat.get('n',site), ness_mps) / trace_norm_ness_mps   ) #
    n_b_exp[site] = np.real( ptn.mp.expectation(purified_id, lat.get('nb',site), ness_mps) / trace_norm_ness_mps   ) #

phonon_rdm_t = np.array(ptn.mp.rdm.o1rdm(ness_mps,4) )
phonon_rdm_t /= np.trace(phonon_rdm_t)

np.savetxt('n_exp', n_exp)
np.savetxt('n_b_exp', n_b_exp)
np.savetxt('phonon_rdm_t',phonon_rdm_t)

print('n_exp = ', n_exp)
print('n_b_exp = ', n_b_exp)
print('phonon_rdm_t = ', phonon_rdm_t)