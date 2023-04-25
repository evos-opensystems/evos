"""Add descr.
"""

import evos.src.lattice.dot_with_oscillator_lattice as lattice 
import evos.src.methods.lindblad as lindblad
import evos.src.methods.partial_traces.partial_trace_tls_boson as pt 

import numpy as np 
#from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as la
import sys
#import math
import os
import argparse

arg_parser = argparse.ArgumentParser(description = "Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with ed lindblad. The dimension of the oscillator needs to be strongly truncated.")
arg_parser.add_argument("-b",   "--bosons", dest = 'max_bosons',  default = 4, type = int, help = 'number of bosonic degrees of freedom - 1 [4]')
arg_parser.add_argument("-dt",   "--timestep", dest = 'dt',  default = 50, type = float, help = 'timestep [50]')
arg_parser.add_argument("-t_max",   "--max_time", dest = 't_max',  default = 5000, type = float, help = 'maximal simulated time [5000]')
arg_parser.add_argument("-mu_l",   "--checmical_pot_left_lead", dest = 'mu_l',  default = +0.5, type = float, help = 'checmical pot. left lead [0.5]')
arg_parser.add_argument("-mu_r",   "--checmical_pot_right_lead", dest = 'mu_r',  default = -0.5, type = float, help = 'checmical pot. right lead [-0.5]')

#FIXME: ADD MU_L AND MU_R
args = arg_parser.parse_args()


np.set_printoptions(threshold=sys.maxsize)
sys.stdout.write('test')

#PARAMETERS
max_bosons = args.max_bosons

om_0 = 0.2 #0.2
m = 1
lamb = 0.1 #0.1
x0 = np.sqrt( 2./ (m * om_0) )
F = 2 *lamb / x0

eps = 0  
Om_kl = +0.5
Om_kr = -0.5
Gamma = 2
g_kl = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
g_kr = np.sqrt( Gamma / (2.*np.pi) ) #FIXME: is this correct?
N0 = 0.5 #0.5,  FIXME: is this correct?
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
writing_dir = make_writing_dir_and_change_to_it('data_lindblad_ed', parameter_dict, overwrite=True)

################


#LATTICE
lat = lattice.DotWithOscillatorLattice(max_bosons)

class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps):
        h_s = eps * lat.sso('ch',1) @ lat.sso('c',1)
        return h_s 

    def h_b(self, Om_kl, Om_kr):
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = Om_kl * lat.sso('ch',0) @ lat.sso('c',0) + Om_kr * lat.sso('ch',3) @ lat.sso('c',3)
        return h_b
   
    def h_t(self, g_kl, g_kr):
        h_t = g_kl * ( lat.sso('ch',1) @ lat.sso('c',0) + lat.sso('ch',0) @ lat.sso('c',1) ) + g_kr * ( lat.sso('ch',1) @ lat.sso('c',3) + lat.sso('ch',3) @ lat.sso('c',1) )
        return h_t
    
    def h_v(self, om_0, F):
        dimH = lat.sso('c',0).shape[0]
        h_v = om_0 * lat.sso('ah',2) @ lat.sso('a',2) - F * ( lat.sso('ch',1) @ lat.sso('c',1) - N0 * np.eye(dimH, dtype='complex') ) @ ( lat.sso('ah',2) + lat.sso('a',2) ) 
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F):
        h_tot = self.h_s(eps) + self.h_b(Om_kl, Om_kr) + self.h_t(g_kl, g_kr) + self.h_v(om_0, F)
        return h_tot
        
    
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)
# h_s = ham.h_s(eps)
# h_b = ham.h_b(Om_kl, Om_kr)
# h_t = ham.h_t(g_kl, g_kr)
# h_v = ham.h_v(om_0, F)
h_tot = ham.h_tot(eps, Om_kl, Om_kr, g_kl, g_kr, om_0, F)


#Lindblad operators
def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

def lindblad_op_list_left_lead( Om_kl, delta_l, mu_l, T_l ):
    l_list_left = []
    l_list_left.append( np.sqrt( delta_l * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l ) ) * lat.sso( 'c',0 ) )
    l_list_left.append( np.sqrt( delta_l * fermi_dist( 1./T_l, Om_kl, mu_l)) * lat.sso('ch',0) )
    return l_list_left

def lindblad_op_list_right_lead( Om_kr, delta_r, mu_r, T_r ):
    l_list_right = []
    l_list_right.append( np.sqrt( delta_r * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) ) * lat.sso( 'c',3 ) )
    l_list_right.append( np.sqrt( delta_r * fermi_dist( 1./T_r, Om_kr, mu_r)) * lat.sso('ch',3) )
    return l_list_right

l_list_left = lindblad_op_list_left_lead( Om_kl, delta_l, mu_l, T_l )
l_list_right = lindblad_op_list_right_lead( Om_kr, delta_r, mu_r, T_r )
l_list = l_list_left + l_list_right

#Initial State: using vacuum for now
init_state = lat.vacuum_state
#NOTE: creating a particle on site 0. used for debugging!
init_state = lat.sso('ch',0) @ init_state #FIXME:remove this!

#Solve Lindblad Equation
lindblad = lindblad.Lindblad(4)
rho_0 = lindblad.ket_to_projector(init_state)        
rho_t = lindblad.solve_lindblad_equation(rho_0, dt, t_max, [], h_tot) #l_list, [], lat.sso('ch',0)

#Compute observables
observables = {'n_system': lat.sso('ch',1) @ lat.sso('c',1), 'U_from_full_state': om_0 * lat.sso('ah',2) @ lat.sso('a',2), 'n_bos':lat.sso('ah',2) @ lat.sso('a',2), 'n_0': lat.sso('ch',0) @ lat.sso('c',0), 'n_3': lat.sso('ch',3) @ lat.sso('c',3)  }
computed_observables =  lindblad.compute_observables(rho_t, observables, dt, t_max)

#compute bosonic reduced density matrix at each timestep
def compute_rho_bosonic(rho0123):
    rho123 = pt.tracing_out_one_tls_from_tls_bosonic_system(0, rho0123, [1,1,0,1], max_bosons)
    rho23 = pt.tracing_out_one_tls_from_tls_bosonic_system(0, rho123, [1,0,1], max_bosons)
    rho2 = pt.tracing_out_one_tls_from_tls_bosonic_system(1, rho23, [0,1], max_bosons)
    return rho2

rho_bosonic = np.zeros( (max_bosons + 1, max_bosons + 1, n_timesteps), dtype='complex')
for time in range(n_timesteps):
    rho_bosonic[:,:,time] = compute_rho_bosonic( rho_t[:,:,time] )
    #print('trace rho_bosonic[:,:,{}] = {}'.format(time, np.trace( rho_bosonic[:,:,time] ) ) )  

##compute non-equilibrium free energy
#bosonic hamiltonian
n_op_bos = range(0, max_bosons + 1)
n_op_bos = np.diag(n_op_bos)
h_bos_reduced = om_0 * n_op_bos 

#compute internal energy
U = np.zeros( n_timesteps )
for time in range(n_timesteps):
    U[time] = np.trace( h_bos_reduced @ rho_bosonic[:,:,time] )

#compute von neuman entropy
def von_neumann_entropy(rho):
    from scipy import linalg as sla
    R = rho*(sla.logm(rho)/sla.logm(np.matrix([[2]])))
    S = -np.matrix.trace(R)
    return(S)

S = np.zeros( n_timesteps )
for time in range(n_timesteps):
    S[time] = von_neumann_entropy( rho_bosonic[:,:,time] )
#non equilibrium free energy
f_neq = U - T_l * S

#equilibrium free energy
Z = np.trace( expm( -h_bos_reduced /(k_b * T_l) ) )
f_eq = -k_b * T_l * np.log(Z)
f_eq_vector = f_eq * np.ones(n_timesteps)

def compute_sec_ord_coherence_funct(rdm):
    numerator = 0.
    denominator = 0.
    for mode in range(max_bosons+1):
        numerator += mode * (mode - 1) * rdm[ mode, mode ]
        denominator += (mode * rdm[ mode, mode ])
    denominator = denominator ** 2
    sec_ord_coherence_funct = numerator/denominator
    return sec_ord_coherence_funct    

sec_ord_coherence_funct_vec = np.zeros(n_timesteps)
for time in range(n_timesteps):
    sec_ord_coherence_funct_vec[time] =  compute_sec_ord_coherence_funct( rho_bosonic[:,:,time] )
    

#PLOT RDM
# fig = plt.figure()
# plt.imshow( np.real( rho_bosonic[:,:,-1] ), aspect='auto' )
# plt.colorbar()
# fig.savefig('lind_phon_rdm.png')
# quit()

#SAVE OBSERVABLES
np.savetxt('n_system', computed_observables['n_system'] )
np.savetxt('U_from_full_state', computed_observables['U_from_full_state'] )
np.savetxt('n_bos', computed_observables['n_bos'] )
np.savetxt('n_0', computed_observables['n_0'] )
np.savetxt('n_3', computed_observables['n_3'] )

np.savetxt('S',S)
np.savetxt('f_neq', f_neq)
np.savetxt('f_eq_vector', f_eq_vector)
np.savetxt('sec_ord_coherence_funct',sec_ord_coherence_funct_vec)
#PLOT
fig = plt.figure()
# plt.plot(time_v, computed_observables['n_0'], label = 'n_0' )
# plt.plot(time_v, computed_observables['n_system'], label = 'n_system' )
# plt.plot(time_v, computed_observables['n_bos'], label = 'n_bos' )
# plt.plot(time_v, computed_observables['n_3'], label = 'n_3' )


#plt.plot(time_v, computed_observables['U_from_full_state'], label = 'U_from_full_state' )
# plt.plot(time_v, rho_bosonic[0,0,:], label = 'occ mode 0 boson' )
# plt.plot(time_v, rho_bosonic[1,1,:], label = 'occ mode 1 boson' )
# plt.plot(time_v, rho_bosonic[2,2,:], label = 'occ mode 2 boson' )
#plt.plot(time_v, rho_bosonic[3,3,:], label = 'occ mode 3 boson' )

# plt.plot(time_v, computed_observables['n_bos'], label = 'n bosons' )


#plt.plot(time_v, U, label = 'U boson' )
#plt.plot(time_v, S, label = 'S boson' )

# plt.plot(time_v, f_neq, label = 'f_neq' )
#plt.plot(time_v, f_neq - f_eq_vector, label = 'W_f' )
# plt.plot(time_v, f_eq_vector, label = 'f_eq')
plt.plot(time_v, sec_ord_coherence_funct_vec, label='g2')

plt.legend()
fig.savefig('ed_lindblad.png')
#plt.show()