
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

n_sites = 2 # number of system sites
n_lead_left = 2 # number of lindblad operators acting on leftest site
n_lead_right = 1 # number of lindblad operators acting on rightmost site

n_tot = n_sites + n_lead_left + n_lead_right
dim_H_sys = 2 ** n_sites
dim_H_lead_left = 2 ** n_lead_left 
dim_H_lead_right = 2 ** n_lead_right
dim_tot = 2 ** n_tot

# temperature and chemical potential on the different leads
T_L = 1
T_R = 1
mu_L = 1
mu_R = -1

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
    
#RIGHT LEAD
eps_step_r = W / n_lead_right
L_discretization_r = int( 2*W/eps_step_r - 1 )
eps_vector_r = np.arange( -W,W, eps_step_r )

eps_delta_vector_r = eps_step_r * np.ones( len( eps_vector_r ) )
k_vector_r = np.zeros( len( eps_vector_r ) )
for i in range( len(eps_vector_r ) ):
    k_vector_r[i] = np.sqrt( const_spec_funct( G ,W, eps_vector_r[i] ) * eps_delta_vector_r[i]/ (2*math.pi) )  
  


#PLOT FITTED LEFT SPECTRAL FUNCTION 
N_points_om = 100 #NOTE: input
om_vector = np.linspace(-W,W,N_points_om)

fitted_spectral_function = np.zeros(N_points_om)
for i in range( 1, len(eps_vector_l)):
    print(i)
    fitted_spectral_function += lorentzian(k_vector_l[i],eps_delta_vector_l[i],eps_vector_l[i], om_vector)


plt.plot(om_vector, G * np.ones(N_points_om), label = 'exact')
plt.plot(om_vector,fitted_spectral_function, label = 'discretized with '+str( len( eps_vector_l ) - 1 ) + ' sites')
plt.legend()
plt.show()
########################################################################################################################


def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

#MAKE LATTICE
spin_lat = spinless_fermions_lattice.SpinlessFermionsLattice(n_tot)

#LEADS HAMILTONIAN

def H_leads_left( eps_vector_l, k_vector_l, mu_L ):
    # LEAD SITES - kinetic energy of left leads
    print(eps_vector_l)
    h_kin_l = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left ): 
        # print( 'on site ', site)
        h_kin_l += ( eps_vector_l[site] - mu_L ) * ( spin_lat.sso('ch',site ) @ spin_lat.sso('c', site ) )
    
    
    
    # # HOPPING BETWEEN LEADS AND SYSTEM LEFT SIDE
    # hop_sys_lead = np.zeros((dim_tot, dim_tot))
    # if n_lead_left == 0: 
    #     print('left sys lead hopping on sites:', 0)
    #     for k in range(0, dim_tot): 
    #         hop_sys_lead = np.zeros((dim_tot, dim_tot))
    # else: 
    #     for k in range(n_lead_left, n_lead_left+1): 
    #         print('left sys lead hopping on sites:', k, k+1)
    #         hop_sys_lead += k_vec[k-1]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
     
        
    # H = kin_leads + hop_sys_lead    
    # return H


H_leads_left =  H_leads_left(eps_vector_l, k_vector_l, mu_L)