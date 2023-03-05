
import evos.src.methods.lindblad as lindblad
import evos.src.lattice.spinless_fermions_lattice as spinless_fermions_lattice

import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as LA
import sys
import math
import os
# from pathlib import Path
# from scipy.optimize import curve_fit

n_system = 1 # number of system sites
n_lead_left = 2 # number of lindblad operators acting on leftest site
n_lead_right = 2 # number of lindblad operators acting on rightmost site
plot_fitted_spectral_density = False
n_tot = n_system + n_lead_left + n_lead_right
dim_H_sys = 2 ** n_system
dim_H_lead_left = 2 ** n_lead_left 
dim_H_lead_right = 2 ** n_lead_right
dim_tot = 2 ** n_tot

# temperature and chemical potential on the different leads
T_L = 1
T_R = 1
mu_L = 1
mu_R = -1

#time-evolution parameters
dt = 0.1
t_max = 10
which_timestep = 0  #FIXME : UPDATE THIS IN TEVO LOOP !!!

#system Hamiltonian parameters
A = 1
om = 0.25

####################################################################################################################

def lorentzian(k,g,eps, om_vector):
    return k**2 * g / ( ( om_vector - eps )**2 + (g/2)**2 )

def const_spec_funct(G,W,eps):
    if eps > -W and eps < W:
        return G
    else:
        return 0

#Fit
#parameters
G = 1 #NOTE: input
W = 8 #NOTE: input

#LEFT LEAD
eps_step_l = 2 * W / ( n_lead_left + 1 )
eps_vector_l = np.arange( -W,W,eps_step_l )

eps_delta_vector_l = eps_step_l * np.ones( len(eps_vector_l) )
k_vector_l = np.zeros( len(eps_vector_l) )
for i in range( len(eps_vector_l) ):
    k_vector_l[i] = np.sqrt( const_spec_funct( G ,W, eps_vector_l[i] ) * eps_delta_vector_l[i]/ (2*math.pi) )  
  
#remove the first element!
eps_delta_vector_l = eps_delta_vector_l[1:]
eps_vector_l = eps_vector_l[1:]
k_vector_l = k_vector_l[1:]

    
#RIGHT LEAD
eps_step_r = 2 * W / ( n_lead_right + 1 )
eps_vector_r = np.arange( -W, W, eps_step_r )

eps_delta_vector_r = eps_step_r * np.ones( len(eps_vector_r) )
k_vector_r = np.zeros( len(eps_vector_r) )
for i in range( len(eps_vector_r) ):
    k_vector_r[i] = np.sqrt( const_spec_funct( G ,W, eps_vector_r[i] ) * eps_delta_vector_r[i]/ (2*math.pi) )  
  
#remove the first element!
eps_delta_vector_r = eps_delta_vector_r[1:]
eps_vector_r = eps_vector_r[1:]
k_vector_r = k_vector_r[1:]
print('k_vector_l.shape', k_vector_l.shape)

#PLOT FITTED LEFT SPECTRAL FUNCTION 
if plot_fitted_spectral_density:
    N_points_om = 100 #NOTE: input
    om_vector = np.linspace(-W,W,N_points_om)

    fitted_spectral_function = np.zeros(N_points_om)
    for i in range(len(eps_vector_l)):
        print(i)
        fitted_spectral_function += lorentzian(k_vector_l[i],eps_delta_vector_l[i],eps_vector_l[i], om_vector)


    plt.plot(om_vector, G * np.ones(N_points_om), label = 'exact')
    plt.plot(om_vector,fitted_spectral_function, label = 'discretized with '+str( len( eps_vector_l )) + ' sites')
    plt.legend()
    plt.show()


def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

#MAKE LATTICE
spin_lat = spinless_fermions_lattice.SpinlessFermionsLattice(n_tot)

#LEADS HAMILTONIAN

def H_leads_left( eps_vector_l, k_vector_l, mu_L ):
    # onsite energy of left leads
    # print(eps_vector_l)
    h_onsite_l = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left ): 
        # print( 'on site ', site)
        h_onsite_l += ( eps_vector_l[site] - mu_L ) * ( spin_lat.sso('ch',site ) @ spin_lat.sso('c', site ) )
    
    # system-lead coupling
    #every lead site is coupled to the leftmost system site
    h_sys_lead = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range(n_lead_left):
        # print( 'coupling {} and {}'.format( site, n_lead_left ) )
        h_sys_lead += k_vector_l[site] *  ( spin_lat.sso('ch',site ) @ spin_lat.sso('c', n_lead_left ) + spin_lat.sso('ch',n_lead_left ) @ spin_lat.sso('c', site ) )
    
        
    h_left_lead = h_onsite_l + h_sys_lead    
    return h_left_lead


def H_leads_right( eps_vector_r, k_vector_r, mu_R ):
    # onsite energy of left leads
    print(eps_vector_r)
    h_onsite_r = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left + n_system , n_tot ): 
        print( 'on site ', site)
        h_onsite_r += ( eps_vector_r[ site - n_lead_left - n_system ] - mu_R ) * ( spin_lat.sso('ch',site ) @ spin_lat.sso('c', site ) )
    
    # system-lead coupling
    #every lead site is coupled to the rightmost system site
    h_sys_lead = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left + n_system , n_tot ):
        print( 'coupling {} and {}'.format( site, n_system + n_lead_left -1 ) )
        h_sys_lead += k_vector_r[ site - n_lead_left - n_system ] *  ( spin_lat.sso('ch',site ) @ spin_lat.sso('c', n_system + n_lead_left -1 ) + spin_lat.sso('ch',n_system + n_lead_left -1 ) @ spin_lat.sso('c', site ) )
    
        
    h_left_lead = h_onsite_r + h_sys_lead    
    return h_left_lead


def lindblad_op_list_left_lead( eps_delta_vector_l, eps_vector_l, mu_L, T_L ):
    l_list_left = []
    for site in range(n_lead_left):
        l_list_left.append( np.sqrt( eps_delta_vector_l[site] * np.exp( 1./T_L * ( eps_vector_l[site] - mu_L ) ) * fermi_dist( 1./T_L, eps_vector_l[site], mu_L ) ) * spin_lat.sso( 'c',site ) )
        l_list_left.append( np.sqrt( eps_delta_vector_l[site]* fermi_dist( 1./T_L, eps_vector_l[site], mu_L)) * spin_lat.sso('ch',site) )
    return l_list_left

def lindblad_op_list_right_lead( eps_delta_vector_r, eps_vector_r, mu_R, T_R ):
    l_list_right = []
    for site in range(n_lead_left):
        l_list_right.append( np.sqrt( eps_delta_vector_r[site] * np.exp( 1./T_R * ( eps_vector_r[site] - mu_R ) ) * fermi_dist( 1./T_R, eps_vector_r[site], mu_R ) ) * spin_lat.sso( 'c',site ) )
        l_list_right.append( np.sqrt( eps_delta_vector_r[site]* fermi_dist( 1./T_R, eps_vector_r[site], mu_R) ) * spin_lat.sso('ch',site) )
    return l_list_right     


def H_sistem_t(A, om, dt, t_max, which_timestep):
    t_vec = np.arange(0,t_max,dt)
    eps = A * np.cos( om * t_vec[which_timestep] )
    h_sys = eps * spin_lat.sso('ch', n_lead_left) @ spin_lat.sso('ch', n_lead_left)
    return h_sys 



#MAKE LEADS
H_leads_left =  H_leads_left(eps_vector_l, k_vector_l, mu_L)
H_leads_right = H_leads_right(eps_vector_r, k_vector_r, mu_R)
l_list_left = lindblad_op_list_left_lead( eps_delta_vector_l, eps_vector_l, mu_L, T_L )
l_list_right = lindblad_op_list_right_lead( eps_delta_vector_r, eps_vector_r, mu_R, T_R )



# H_sistem_t = H_sistem_t(A, om, dt, t_max, which_timestep) #FIXME: build it at every timestep!
######