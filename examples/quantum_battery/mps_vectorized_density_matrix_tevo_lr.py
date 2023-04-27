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
ptn.env.setDisableBacktraces(True)
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

T_l = 1./0.5 #beta_l = 0.5
T_r = 1./0.5 #beta_r = 0.5
k_b = 1 #boltzmann constant
 
dt = args.dt
t_max = args.t_max
time_v = np.arange(0, t_max, dt)
n_timesteps = int(t_max/dt)
n_trajectories = 1
first_trajectory = 0

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

parameter_dict = {'max_bosons': max_bosons, 'dt': dt, 't_max': t_max, 'mu_l' : mu_l, 'mu_r' : mu_r  }
writing_dir = make_writing_dir_and_change_to_it('data_mps_lindblad', parameter_dict, overwrite=True)

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
    
for site in range(10):
    #print('<n> on site {} is {}'.format(site, ptn.mp.expectation(vac_state, lat.get('n',site) ) ) )
    print('<n_b> on site {} is {}'.format(site, ptn.mp.expectation(vac_state, lat.get('nb',site) ) ) )
    

#Hamiltonian
class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps, even_odd_idx_shift): #system
        h_s = eps * lat.get('n',2 + even_odd_idx_shift)  
        print('h_s on site ', 2 + even_odd_idx_shift)
        return h_s 

    def h_b(self, Om_kl, Om_kr, even_odd_idx_shift): #leads
        #NOTE: added mu_l and mu_rto onsite energies
        h_b =  Om_kl * lat.get('n',0 + even_odd_idx_shift) +  Om_kr * lat.get('n',8 + even_odd_idx_shift)
        print('h_b on sites {} and {} '.format(even_odd_idx_shift, 8+even_odd_idx_shift))
        return h_b
   
    def h_t(self, g_kl, g_kr, even_odd_idx_shift): #system-leads
        
        h_t = g_kl * ( lat.get('c',2 + even_odd_idx_shift) * lat.get('ch',0 + even_odd_idx_shift) + lat.get('c',0 + even_odd_idx_shift) * lat.get('ch',2 + even_odd_idx_shift) ) 
        h_t += g_kr * ( lat.get('c',2 + even_odd_idx_shift) * lat.get('ch',8 + even_odd_idx_shift) + lat.get('c',8 + even_odd_idx_shift) * lat.get('ch',2 + even_odd_idx_shift) ) 
        print('left hopping between {} and {}'.format(even_odd_idx_shift, 2 + even_odd_idx_shift))
        print('left hopping between {} and {}'.format(8 + even_odd_idx_shift, 2 + even_odd_idx_shift))
        return h_t
    
    def h_boson(self, om_0, even_odd_idx_shift): #oscillator
        h_boson = om_0 * lat.get('nb',4 + 2*even_odd_idx_shift) 
        print('h_boson on site {}'.format(4 + 2*even_odd_idx_shift))
        return h_boson
    
    def h_v(self, F, N0, even_odd_idx_shift): #system-oscillator
        h_v = - F * ( lat.get('n',2 + even_odd_idx_shift) - N0 * lat.get('I') ) *  ( lat.get('ah',4 + 2*even_odd_idx_shift) * lat.get('a',5 + 2*even_odd_idx_shift)  + lat.get('a',4 + 2*even_odd_idx_shift) * lat.get('ah',5 + 2*even_odd_idx_shift) ) 
        print('h_v between system-tls {} and physical {} and pp {} sites'. format(2 + even_odd_idx_shift,4 + 2*even_odd_idx_shift, 5 + 2*even_odd_idx_shift ))
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, even_odd_idx_shift=0):
        print('even_odd_idx_shift = ', even_odd_idx_shift)
        h_tot =  self.h_s(eps, even_odd_idx_shift) + self.h_b(Om_kl, Om_kr, even_odd_idx_shift) + self.h_t(g_kl, g_kr, even_odd_idx_shift) + self.h_boson(om_0, even_odd_idx_shift) + self.h_v(F, N0, even_odd_idx_shift) 
        #h_tot.truncate()
        return h_tot    
        
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)      
h_tot_even = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, even_odd_idx_shift = 0)
h_tot_odd = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F, even_odd_idx_shift = even_odd_idx_shift)

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
        op *= lat.get('a',site + 1 ) * lat.get('ah',site)  *  lat.get('a',site + 2*even_odd_idx_shift + 1) * lat.get('ah',site + 2*even_odd_idx_shift) #* 1./mode #FIXME: reverse order?
        op *= 1./mode
        op_tot += op
        op.truncate()
        op_tot.truncate()
        
    return op_tot   

purified_id = vac_state.copy()     
for site in [0,2,8]:
    purified_id *=  mpo_max_ent_pair_ferm(site)

for site in [4]:    
    purified_id *= mpo_max_ent_pair_bos(site, max_bosons)            
    

# for site in range(10):
#     print('<n> on site {} is {}'.format(site, ptn.mp.expectation(purified_id, lat.get('n',site) ) ) )
#     #print('<n_b> on site {} is {}'.format(site, ptn.mp.expectation(purified_id, lat.get('nb',site) ) ) )
# quit()        


#NON-HERMITIAN, IMAGINARY-TIME TDVP CONFIG 
conf_tdvp = ptn.tdvp.Conf()
conf_tdvp.mode = ptn.tdvp.Mode.GSE   #TwoSite, GSE, Subspace
conf_tdvp.dt = 1j * dt
conf_tdvp.trunc.threshold = 1e-8  #NOTE: set to zero for gse
conf_tdvp.trunc.weight = 1e-10 #tdvp_trunc_weight #NOTE: set to zero for gse
conf_tdvp.trunc.maxStates = 3000
conf_tdvp.exp_conf.errTolerance = 1e-7
conf_tdvp.exp_conf.inxTolerance = 1e-6
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

#INITIALIZE OBSERVABLES
n_exp = np.zeros( ( 10, n_timesteps) )
n_b_exp = np.zeros( ( 10, n_timesteps) )
phys_dim_phon = np.zeros( (n_timesteps) )
bond_dim =  np.zeros( ( 10, n_timesteps) )
phonon_rdm = np.zeros( (max_bosons +1, max_bosons +1, n_timesteps), dtype='complex' )
#FIXME: ONLY FOR DEBUGGING: excite particle on sites 0,1
# vac_state *= lat.get('ch',0)    
# vac_state *= lat.get('ch',1)  
psi_t = vac_state.copy()
#normalize initial state
psi_t.normalise()

#main tevo loop
for time in range(n_timesteps):
    #reinitialize worker with normalized state
    worker = ptn.mp.tdvp.PTDVP( psi_t.copy(),[vectorized_lindbladian.copy()],conf_tdvp.copy() ) 
    worker.do_step()
    psi_t = worker.get_psi(False)
    
    #Compute trace-norm for observables
    trace_norm_psi_t = ptn.mp.overlap(purified_id, psi_t)
    
    #compute observables dividing by trace-norm
    for site in range(10):
        n_exp[site, time] = np.real( ptn.mp.expectation(purified_id, lat.get('n',site), psi_t) / trace_norm_psi_t   ) #
        n_b_exp[site, time] = np.real( ptn.mp.expectation(purified_id, lat.get('nb',site), psi_t) / trace_norm_psi_t   ) #
        bond_dim[site, time] = psi_t[site].getTotalDims()[2]
            
    phys_dim_phon[time] = psi_t[4].getTotalDims()[0]        
    phonon_rdm_t = np.array(ptn.mp.rdm.o1rdm(psi_t,4) )
    phonon_rdm_t /= np.trace(phonon_rdm_t)
    phonon_rdm[ :phonon_rdm_t.shape[0], :phonon_rdm_t.shape[1], time ] = phonon_rdm_t 
    #save observables
    np.save('n_exp', n_exp )
    np.save('n_b_exp', n_b_exp )
    np.save('phonon_rdm',phonon_rdm)
    np.save('bond_dim',bond_dim)
    np.save('phys_dim_phon',phys_dim_phon)

    #Normalize state to reinitialize tdvp worker
    psi_t.normalise()
    
    
#PLOT
# time_v = np.linspace(0,t_max,n_timesteps)

# fig = plt.figure()

# plt.plot(time_v, n_exp[0,:], label='n0')
# plt.plot(time_v, n_exp[1,:], label='n1')
# plt.plot(time_v, n_b_exp[4,:], label='nb4')
# plt.plot(time_v, n_exp[8,:], label='n8')
# #plt.plot(time_v, phys_dim_phon[:], label='phys_dim_phon')

# plt.legend()
# fig.savefig('mps_vectorized_density_matrix_tevo_lr.png')    

