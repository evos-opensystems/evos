"""Trying to reproduce 'https://arxiv.org/pdf/2201.07819.pdf' for a single driven left lead, and a single driven right one with ed lindblad. The dimension of the oscillator needs to be strongly truncated.
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
np.set_printoptions(threshold=sys.maxsize)

#PARAMETERS
max_bosons = 2
eps = 1
Om_kl = 1
Om_kr = 1
g_kl = 1
g_kr = 1
om_0 = 1
F = 1
#FIXME: how do I compute N0??

Gamma = 0.5
mu_l = +1
mu_r = +1
T_l = 1
T_r = 1
k_b = 1 #boltzmann constant
 
dt = 1
t_max = 100
time_v = np.arange(0, t_max, dt)
n_timesteps = int(t_max/dt)

os.chdir('data_lindblad_ed')
#LATTICE
lat = lattice.DotWithOscillatorLattice(max_bosons)

class Hamiltonian():
    
    def __init__(self, lat, max_bosons):
        self.lat = lat
        self.max_bosons = max_bosons
        
    def h_s(self, eps):
        
        h_s = eps * lat.sso('ch',1) @ lat.sso('c',1)
        return h_s 

    def h_b(self, Om_kl, Om_kr, mu_l, mu_r):
        #NOTE: added mu_l and mu_rto onsite energies
        h_b = ( Om_kl - mu_l ) * lat.sso('ch',0) @ lat.sso('c',0) + ( Om_kr - mu_r ) * lat.sso('ch',3) @ lat.sso('c',3)
        return h_b
   
    def h_t(self, g_kl, g_kr):
        h_t = g_kl * ( lat.sso('ch',1) @ lat.sso('c',0) + lat.sso('ch',0) @ lat.sso('c',1) ) + g_kr * ( lat.sso('ch',1) @ lat.sso('c',3) + lat.sso('ch',3) @ lat.sso('c',1) )
        return h_t
    
    def h_v(self, om_0, F):
        #FIXME: need to detract N0
        #FIXME: is m = 1 ?
        h_v = om_0 * lat.sso('ah',2) @ lat.sso('a',2) - F * lat.sso('ch',1) @ lat.sso('c',1) @ ( lat.sso('ah',2) + lat.sso('a',2) ) 
        return h_v 
    
    def h_tot(self, eps, Om_kl, Om_kr, mu_l, mu_r, g_kl, g_kr, om_0, F):
        h_tot = self.h_s(eps) + self.h_b(Om_kl, Om_kr, mu_l, mu_r) + self.h_t(g_kl, g_kr) + self.h_v(om_0, F)
        return h_tot
        
    
#Hamiltonian
ham = Hamiltonian(lat, max_bosons)
# h_s = ham.h_s(eps)
# h_b = ham.h_b(Om_kl, Om_kr, mu_l, mu_r)
# h_t = ham.h_t(g_kl, g_kr)
# h_v = ham.h_v(om_0, F)
h_tot = ham.h_tot(eps, Om_kl, Om_kr, mu_l, mu_r, g_kl, g_kr, om_0, F)


#Lindblad operators
def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

def lindblad_op_list_left_lead( Om_kl, Gamma, mu_l, T_l ):
    l_list_left = []
    l_list_left.append( np.sqrt( Gamma * np.exp( 1./T_l * ( Om_kl - mu_l ) ) * fermi_dist( 1./T_l, Om_kl, mu_l ) ) * lat.sso( 'c',0 ) )
    l_list_left.append( np.sqrt( Gamma * fermi_dist( 1./T_l, Om_kl, mu_l)) * lat.sso('ch',0) )
    return l_list_left

def lindblad_op_list_right_lead( Om_kr, Gamma, mu_r, T_r ):
    l_list_right = []
    l_list_right.append( np.sqrt( Gamma * np.exp( 1./T_r * ( Om_kr - mu_r ) ) * fermi_dist( 1./T_r, Om_kr, mu_r ) ) * lat.sso( 'c',3 ) )
    l_list_right.append( np.sqrt( Gamma * fermi_dist( 1./T_r, Om_kr, mu_r)) * lat.sso('ch',3) )
    return l_list_right

l_list_left = lindblad_op_list_left_lead( Om_kl, Gamma, mu_l, T_l )
l_list_right = lindblad_op_list_right_lead( Om_kr, Gamma, mu_r, T_r )
l_list = l_list_left + l_list_right

#Initial State: using vacuum for now
#NOTE: vacuum for leads (compare with ed qj) or thermal state on leads (compare with doubled qj?
init_state = lat.vacuum_state
#

#Solve Lindblad Equation
lindblad = lindblad.Lindblad(4)
#init_state = lat.sso('ch',1) @ init_state #FIXME: remove this!!!!!!! 
rho_0 = lindblad.ket_to_projector(init_state)        
rho_t = lindblad.solve_lindblad_equation(rho_0, dt, t_max,l_list, h_tot) #l_list

#Compute observables
observables = {'n_system': lat.sso('ch',1) @ lat.sso('c',1), 'U_from_full_state': om_0 * lat.sso('ah',2) @ lat.sso('a',2) }
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

#SAVE OBSERVABLES
np.savetxt('n_system', computed_observables['n_system'] )
np.savetxt('U_from_full_state', computed_observables['U_from_full_state'] )
np.savetxt('S',S)
#PLOT
fig = plt.figure()
#plt.plot(time_v, computed_observables['n_system'], label = 'n_system' )
#plt.plot(time_v, computed_observables['U_from_full_state'], label = 'U_from_full_state' )
plt.plot(time_v, rho_bosonic[0,0,:], label = 'occ mode 0 boson' )
plt.plot(time_v, rho_bosonic[1,1,:], label = 'occ mode 1 boson' )
plt.plot(time_v, rho_bosonic[2,2,:], label = 'occ mode 2 boson' )
##plt.plot(time_v, rho_bosonic[3,3,:], label = 'occ mode 3 boson' )

#plt.plot(time_v, U, label = 'U boson' )
#plt.plot(time_v, S, label = 'S boson' )

# plt.plot(time_v, f_neq, label = 'f_neq' )
#plt.plot(time_v, f_neq - f_eq_vector, label = 'W_f' )
# plt.plot(time_v, f_eq_vector, label = 'f_eq')
plt.legend()
fig.savefig('lindblad.png')
#plt.show()