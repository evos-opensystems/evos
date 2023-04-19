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
N0 = 0. #FIXME: is this correct?
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

idx_shift_lattice_doubling = 5

os.chdir('data_mps_lindblad')

#Lattice
ferm_bos_sites = [0,0,1,0,  0,0,1,0]
lat = ptn.mp.lat.u1.genSpinlessFermiBose_NilxU1( ferm_bos_sites, max_bosons)
lat = ptn.mp.proj_pur.proj_purification(lat, [1], ["a", "ah"])
#print(lat)

#FIXME: PP vacuum is wrong!!
vac_state =  ptn.mp.proj_pur.generateNearVacuumState(lat, 2, "0," + str( max_bosons ) )

#creating PP vacuum
for mode in range(max_bosons):
    vac_state *= lat.get('ah',3)
    vac_state.normalise()
    vac_state *= lat.get('ah',8)
    vac_state.normalise()

#destroy particles on fermionic sites
for site in [0,1,4,5,6,9]:
    vac_state *= lat.get('c',site)
    vac_state.normalise()




# for site in range(10):
#     #print('<n> on site {} is {}'.format(site, ptn.mp.expectation(vac_state, lat.get('n',site) ) ) )
#     print('<n_b> on site {} is {}'.format(site, ptn.mp.expectation(vac_state, lat.get('nb',site) ) ) )
# quit()

#Hamiltonian
class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps, idx_shift_lattice_doubling): #system
        h_s = eps * lat.get('n',1 + idx_shift_lattice_doubling)  #no need to PP
        #h_s = ptn.mp.addLog(h_s)
        #h_s.truncate()
        return h_s 

    def h_b(self, Om_kl, Om_kr, idx_shift_lattice_doubling): #leads
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = []
        h_b.append( Om_kl * lat.get('n',0 + idx_shift_lattice_doubling) ) #no need to PP
        h_b.append( Om_kr * lat.get('n',4 + idx_shift_lattice_doubling) ) #no need to PP
        h_b = ptn.mp.addLog(h_b)
        #h_b.truncate()
        return h_b
   
    def h_t(self, g_kl, g_kr, idx_shift_lattice_doubling): #system-leads
        
        h_t = g_kl * ( lat.get('c',1 + idx_shift_lattice_doubling) * lat.get('ch',0 + idx_shift_lattice_doubling) + lat.get('c',0 + idx_shift_lattice_doubling) * lat.get('ch',1 + idx_shift_lattice_doubling) ) 
        h_t += g_kr * ( lat.get('c',1 + idx_shift_lattice_doubling) * lat.get('ch',4 + idx_shift_lattice_doubling) + lat.get('c',4 + idx_shift_lattice_doubling) * lat.get('ch',1 + idx_shift_lattice_doubling) ) 
    
        #h_t.truncate()
        return h_t
    
    def h_boson(self, om_0, idx_shift_lattice_doubling): #oscillator
        h_boson = om_0 * lat.get('nb',2 + idx_shift_lattice_doubling) 
        return h_boson
    
    def h_v(self, F, N0, idx_shift_lattice_doubling): #system-oscillator
        h_v = - F * ( lat.get('n',1 + idx_shift_lattice_doubling) - N0 * lat.get('I') ) *  ( lat.get('ah',2 + idx_shift_lattice_doubling) * lat.get('a',3 + idx_shift_lattice_doubling)  + lat.get('a',2 + idx_shift_lattice_doubling) * lat.get('ah',3 + idx_shift_lattice_doubling) ) 
        #h_v.truncate()
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, idx_shift_lattice_doubling=0):
        print('idx_shift_lattice_doubling = ', idx_shift_lattice_doubling)
        h_tot =  self.h_s(eps, idx_shift_lattice_doubling) + self.h_t(g_kl, g_kr, idx_shift_lattice_doubling) + self.h_b(Om_kl, Om_kr, idx_shift_lattice_doubling)  + self.h_boson(om_0, idx_shift_lattice_doubling) + self.h_v(F, N0, idx_shift_lattice_doubling) 
        #h_tot.truncate()
        return h_tot    
    
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)      
#h_s_left =  ham.h_v(F, N0, idx_shift_lattice_doubling=5)
h_tot_left = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, idx_shift_lattice_doubling=0)
h_tot_right = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, idx_shift_lattice_doubling = idx_shift_lattice_doubling)
    
# lat.add('h_tot_right', 'h_tot_right', h_tot_right)
# lat.save('lat')
# quit()

#VECTORIZED LINDBLADIAN
vectorized_lindbladian = -1j*h_tot_left + 1j*h_tot_right 

#BUILD UNNORMALIZED PURIFIED IDENTITY
def mpo_max_ent_pair_ferm(site):
    """_summary_
    """
    
    op = lat.get('I')
    op_tot = op.copy()
    for mode in range(1,2):
        #print('mode = ', mode)
        op *= lat.get('ch',site) * lat.get('ch',site + idx_shift_lattice_doubling) 
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
        #print('mode = ', mode)
        op *= lat.get('a',site + 1 ) * lat.get('ah',site)  *  lat.get('a',site + idx_shift_lattice_doubling + 1) * lat.get('ah',site + idx_shift_lattice_doubling) #* 1./mode #FIXME: reverse order?
        op *= 1./mode
        op_tot += op
        op.truncate()
        op_tot.truncate()
        
    return op_tot   


purified_id = vac_state.copy()     
for site in [0,1,4]:
    purified_id *=  mpo_max_ent_pair_ferm(site)

for site in [2]:    
    purified_id *= mpo_max_ent_pair_bos(site, max_bosons)     
    
##purified_id.normalise()
# for site in range(10):
#     #print('<n> on site {} is {}'.format(site, ptn.mp.expectation(purified_id, lat.get('n',site) ) ) )
#     print('<n_b> on site {} is {}'.format(site, ptn.mp.expectation(purified_id, lat.get('nb',site) ) ) )
# quit()    

#NON-HERMITIAN, IMAGINARY-TIME TDVP CONFIG 
 
conf_tdvp = ptn.tdvp.Conf()
conf_tdvp.mode = ptn.tdvp.Mode.GSE   #TwoSite, GSE, Subspace
conf_tdvp.dt = 1j*dt
conf_tdvp.trunc.threshold = 1e-10  #NOTE: set to zero for gse
conf_tdvp.trunc.weight = 1e-15 #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp.trunc.maxStates = 3000
conf_tdvp.exp_conf.errTolerance = 1e-10
conf_tdvp.exp_conf.inxTolerance = 1e-15
conf_tdvp.exp_conf.maxIter =  10
conf_tdvp.cache = 1
conf_tdvp.maxt = 1j*t_max

conf_tdvp.gse_conf.mode = ptn.tdvp.GSEMode.BeforeTDVP
conf_tdvp.gse_conf.krylov_order = 3 #FIXME 3,5 INCRESE
conf_tdvp.gse_conf.trunc_op = ptn.Truncation(1e-8 , maxStates=500) #maxStates shuld be the same as the one used for tdvp! 1e-8 - 1e-6
conf_tdvp.gse_conf.trunc_expansion = ptn.Truncation(1e-6, maxStates=500) #precision of GSE. par is trunc. treshold. do not goe below 10^-12 (numerical instability)!!
conf_tdvp.gse_conf.adaptive = True
conf_tdvp.gse_conf.sing_val_thresholds = [1e-12] # [1e-12] #most highly model-dependet parameter 

conf_tdvp.exp_conf.mode = 'N'  #FIXME: specify this before
conf_tdvp.exp_conf.submode = 'a' #FIXME: specify this before
conf_tdvp.exp_conf.minIter = 20


n_exp = np.zeros( ( 5, n_timesteps) )
n_b_exp = np.zeros( ( 5, n_timesteps) )

#main tevo loop

#excite particle on sites 0, 5
vac_state *= lat.get('ch',0)    
vac_state *= lat.get('ch',5)  
psi_t = vac_state.copy()

for time in range(n_timesteps):
    #reinitialize worker with normalized state
    worker = ptn.mp.tdvp.PTDVP( psi_t.copy(),[ vectorized_lindbladian.copy()], conf_tdvp.copy() ) 
    worker.do_step()
    psi_t = worker.get_psi(False)
    #Compute trace-norm for observables
    trace_norm_psi_t = ptn.mp.overlap(purified_id, psi_t)
    #print(trace_norm_psi_t)
    #quit()
    #compute observables dividing by trace-norm
    for site in range(3):
        n_exp[site,time] = np.real( ptn.mp.expectation(purified_id, lat.get('n',site), psi_t) / trace_norm_psi_t   ) #
        n_b_exp[site,time] = np.real( ptn.mp.expectation(purified_id, lat.get('nb',site), psi_t) / trace_norm_psi_t   ) #

    np.savetxt('n_exp', n_exp )
    np.savetxt('n_b_exp', n_exp )

    #Normalize state to reinitialize tdvp worker
    psi_t.normalise()
    
    
#PLOT
time_v = np.linspace(0,t_max,n_timesteps)

fig = plt.figure()

plt.plot(time_v, n_exp[0,:], label='n0')
plt.plot(time_v, n_exp[1,:], label='n1')
plt.plot(time_v, n_b_exp[2,:], label='nb2')
plt.legend()
fig.savefig('ps_vectorized_density_matrix_tevo_new_lattice2.png')    