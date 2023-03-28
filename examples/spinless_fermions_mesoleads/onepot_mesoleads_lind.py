import evos.src.lattice.spinless_fermions_lattice as spinless_fermions_lattice
import evos.src.methods.ed_time_dep_hamiltonian_lindblad_solver_new as lindblad

import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as la
import sys
import math
import os
np.set_printoptions(threshold=sys.maxsize)

#Parameters
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
mu_L = 0.5
mu_R = - 0.5

#time-evolution parameters
dt = 0.05
t_max = 20
n_timesteps = int(t_max/dt)
which_timestep = 0  #FIXME : UPDATE THIS IN TEVO LOOP !!!

#system Hamiltonian parameters
A = 1
om = 0.25
# spectral density parameters
G = 1 
W = 8 
####################################################################################################################
#Fit

def lorentzian(k,g,eps, om_vector):
    return k**2 * g / ( ( om_vector - eps )**2 + (g/2)**2 )

def const_spec_funct(G,W,eps):
    if eps > -W and eps < W:
        return G
    else:
        return 0

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
# k_vector_l[:] = 0 #FIXME remove it

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
# k_vector_r[:] = 0 #FIXME remove it

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
ferm_lat = spinless_fermions_lattice.SpinlessFermionsLattice(n_tot)

#LEADS HAMILTONIAN

################### FIXME: this is used only to compute observables
def h_onsite_l( eps_vector_l, k_vector_l, mu_L ):
    # onsite energy of left leads
    # print(eps_vector_l)
    h_onsite_l = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left ): 
        # print( 'on site ', site)
        h_onsite_l += ( eps_vector_l[site] - mu_L ) * ( ferm_lat.sso('ch',site ) @ ferm_lat.sso('c', site ) )
    
    return h_onsite_l

h_onsite_l = h_onsite_l( eps_vector_l, k_vector_l, mu_L )
###################

def H_leads_left( eps_vector_l, k_vector_l, mu_L ):
    # onsite energy of left leads
    # print(eps_vector_l)
    h_onsite_l = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left ): 
        # print( 'on site ', site)
        h_onsite_l += ( eps_vector_l[site] - mu_L ) * ( ferm_lat.sso('ch',site ) @ ferm_lat.sso('c', site ) )
    
    # system-lead coupling
    #every lead site is coupled to the leftmost system site
    h_sys_lead = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range(n_lead_left):
        # print( 'coupling {} and {}'.format( site, n_lead_left ) )
        h_sys_lead += k_vector_l[site] *  ( ferm_lat.sso('ch',site ) @ ferm_lat.sso('c', n_lead_left ) + ferm_lat.sso('ch',n_lead_left ) @ ferm_lat.sso('c', site ) )
    
        
    h_left_lead = h_onsite_l + h_sys_lead    
    return h_left_lead


def H_leads_right( eps_vector_r, k_vector_r, mu_R ):
    # onsite energy of left leads
    # print(eps_vector_r)
    h_onsite_r = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left + n_system , n_tot ): 
        # print( 'on site ', site)
        h_onsite_r += ( eps_vector_r[ site - n_lead_left - n_system ] - mu_R ) * ( ferm_lat.sso('ch',site ) @ ferm_lat.sso('c', site ) )
    
    # system-lead coupling
    #every lead site is coupled to the rightmost system site
    h_sys_lead = np.zeros( (dim_tot, dim_tot), dtype = 'complex' )
    for site in range( n_lead_left + n_system , n_tot ):
        # print( 'coupling {} and {}'.format( site, n_system + n_lead_left -1 ) )
        h_sys_lead += k_vector_r[ site - n_lead_left - n_system ] *  ( ferm_lat.sso('ch',site ) @ ferm_lat.sso('c', n_system + n_lead_left -1 ) + ferm_lat.sso('ch',n_system + n_lead_left -1 ) @ ferm_lat.sso('c', site ) )
    
        
    h_left_lead = h_onsite_r + h_sys_lead    
    return h_left_lead

#LEINDBLAD OPERATORS LISTS FOR LEADS
def lindblad_op_list_left_lead( eps_delta_vector_l, eps_vector_l, mu_L, T_L ):
    l_list_left = []
    for site in range(n_lead_left):
        l_list_left.append( np.sqrt( eps_delta_vector_l[site] * np.exp( 1./T_L * ( eps_vector_l[site] - mu_L ) ) * fermi_dist( 1./T_L, eps_vector_l[site], mu_L ) ) * ferm_lat.sso( 'c',site ) )
        l_list_left.append( np.sqrt( eps_delta_vector_l[site]* fermi_dist( 1./T_L, eps_vector_l[site], mu_L)) * ferm_lat.sso('ch',site) )
    return l_list_left

def lindblad_op_list_right_lead( eps_delta_vector_r, eps_vector_r, mu_R, T_R ):
    l_list_right = []
    for site in range(n_lead_right):
        l_list_right.append( np.sqrt( eps_delta_vector_r[site] * np.exp( 1./T_R * ( eps_vector_r[site] - mu_R ) ) * fermi_dist( 1./T_R, eps_vector_r[site], mu_R ) ) * ferm_lat.sso( 'c', n_lead_left + n_system + site ) )
        l_list_right.append( np.sqrt( eps_delta_vector_r[site]* fermi_dist( 1./T_R, eps_vector_r[site], mu_R) ) * ferm_lat.sso('ch', n_lead_left + n_system + site) )
    return l_list_right     


#MAKE LEADS
H_leads_left =  H_leads_left(eps_vector_l, k_vector_l, mu_L)
H_leads_right = H_leads_right(eps_vector_r, k_vector_r, mu_R)

l_list_left = lindblad_op_list_left_lead( eps_delta_vector_l, eps_vector_l, mu_L, T_L )
l_list_right = lindblad_op_list_right_lead( eps_delta_vector_r, eps_vector_r, mu_R, T_R )
l_list_tot = l_list_left + l_list_right 


def H_sistem_t(A, om, dt, t_max, t ): 
    t_vec = np.arange(0,t_max,dt)
    eps = A * np.cos( om * t )  #FIXME: change 0 with t
    h_sys = eps * ferm_lat.sso('ch', n_lead_left) @ ferm_lat.sso('c', n_lead_left)
    return h_sys 


def H_tot_t(t):
    #NOTE: this MUST a function of t only in order to be compatible with the time-dependent lindblad solver
    
    return H_leads_left + H_leads_right + H_sistem_t(A, om, dt, t_max, t ) 




# H_sistem_t = H_sistem_t(A, om, dt, t_max, which_timestep) #FIXME: build it at every timestep!
######

#INITAL STATE: vacuum for leads and 1/sqrt(2) (|0> + |1>) for the system (single dot)
vac = ferm_lat.vacuum_state #vacuum state = all up
sys_up_state = ferm_lat.sso('ch',n_lead_left) @ vac.copy() 
init_state = (vac + sys_up_state)
init_state /= la.norm(init_state)

# print( np.conjugate(init_state)@ ferm_lat.sso('ch',n_lead_left) @ ferm_lat.sso('c',n_lead_left) @  init_state   )

#left lead particle current operator
j_p_left_op = np.zeros( (dim_tot, dim_tot), dtype='complex' )
for site in range(n_lead_left):
    j_p_left_op += k_vector_l[site] * ferm_lat.sso('ch',n_lead_left) @ ferm_lat.sso('c',site)
    j_p_left_op -= k_vector_l[site] * ferm_lat.sso('ch',site) @ ferm_lat.sso('c',n_lead_left)
j_p_left_op *= 1j
#print( np.allclose(j_p_left_op, j_p_left_op.T, rtol=1e-5, atol=1e-8) )
#for site in range(n_lead_left):
# print(j_p_left_op )
# quit()    

#SOLVE LINDBLAD EQUATION WITH REKA'S SOLVER
lindblad = lindblad.SolveLindbladEquation(dim_tot, H_tot_t, l_list_tot, dt, t_max)
# nf_sys, time_v = lindblad.solve( ferm_lat.sso('ch',n_lead_left) @ ferm_lat.sso('c',n_lead_left), init_state ) #UPDATE [L], H at each timestep
#FIXME:works only for n_leads_left = n_leads_right = 2!
# nf_0, time_v = lindblad.solve( ferm_lat.sso('ch',0) @ ferm_lat.sso('c',0), init_state ) #UPDATE [L], H at each timestep
# nf_1, time_v = lindblad.solve( ferm_lat.sso('ch',1) @ ferm_lat.sso('c',1), init_state ) #UPDATE [L], H at each timestep
#nf_2, time_v = lindblad.solve( ferm_lat.sso('ch',2) @ ferm_lat.sso('c',2), init_state ) #UPDATE [L], H at each timestep
# nf_3, time_v = lindblad.solve( ferm_lat.sso('ch',3) @ ferm_lat.sso('c',3), init_state ) #UPDATE [L], H at each timestep
# nf_4, time_v = lindblad.solve( ferm_lat.sso('ch',4) @ ferm_lat.sso('c',4), init_state ) #UPDATE [L], H at each timestep
# curr_L , time_v = lindblad.solve( 0.5j*( ferm_lat.sso('ch',n_lead_left) @ ferm_lat.sso('c',n_lead_left-1) - ferm_lat.sso('ch',n_lead_left-1) @ ferm_lat.sso('c',n_lead_left) ), init_state ) #UPDATE [L], H at each timestep
#en_curr, time_v =  lindblad.solve( h_onsite_l, init_state ) #UPDATE [L], H at each timestep

nf_system, time_v = lindblad.solve( ferm_lat.sso('ch',n_lead_left) @ ferm_lat.sso('c',n_lead_left), init_state ) #UPDATE [L], H at each timestep

#j_p_left, time_v = lindblad.solve( j_p_left_op, init_state ) #UPDATE [L], H at each timestep


# derivative of number of particles on left lead
# N_L = np.array(nf_0[1:]) + np.array(nf_1[1:])
# nf_system_der = np.zeros(len(nf_system))
# for i in range(1,len(nf_system)):
#     nf_system_der[i] = (nf_system[i] - nf_system[i-1])/dt


def thermal_occupation(beta, energy, mu):
    rho = expm( -beta * ( energy - mu ) * np.array( [ [0,0],[0,1] ] ) ) / np.trace( expm( -beta * ( energy - mu ) * np.array( [ [0,0],[0,1] ] ) ) )
    occupation = np.trace( rho @ np.array( [ [0,0],[0,1] ] ) ) 
    return occupation 

# therm_occ_left_0 = thermal_occupation( 1./T_L ,eps_vector_l[0], mu_L ) 
# therm_occ_left_1 = thermal_occupation( 1./T_L ,eps_vector_l[1], mu_L ) 

# therm_occ_right_0 = thermal_occupation( 1./T_R ,eps_vector_r[0], mu_R ) 
# therm_occ_right_1 = thermal_occupation( 1./T_R ,eps_vector_r[1], mu_R ) 


#PLOT
fig = plt.figure()

# plt.plot( time_v[2:], N_L_der[1:] )
# plt.plot( time_v[2:], N_L[1:] )
# plt.plot( time_v[1:], curr_L[1:] )
#LEADS
#left
# plt.plot( time_v[1:], nf_0[1:], label='site 0' )
# plt.plot( time_v[1:], nf_1[1:], label='site 1' )
# #system
plt.plot( time_v[1:], nf_system[1:], label='system site' ) 
# #right
# plt.plot( time_v[1:], nf_3[1:], label='site 3' )
# plt.plot( time_v[1:], nf_4[1:], label='site 4' ) 
#left lead energy current
# plt.plot( time_v[1:], en_curr[1:], label='en curr l' ) 
# #exact leads thermalization
# plt.hlines(y=therm_occ_left_0, xmin=0, xmax = t_max , label='site 0 therm' )
# plt.hlines(y=therm_occ_left_1, xmin=0, xmax = t_max , label='site 1 therm')
# plt.hlines(y=therm_occ_right_0, xmin=0, xmax = t_max , label='site 3 therm')
# plt.hlines(y=therm_occ_right_1, xmin=0, xmax = t_max , label='site 4 therm')
#current
## plt.plot( time_v[50:], nf_system_der[50:], label='system site curr' ) 
#plt.plot( time_v[1:], j_p_left[1:], label='system site curr' ) 

#LOAD QJ DATA
#os.chdir('qj_n_tot5')
os.chdir('qj_n_tot5')
n_system = np.loadtxt('n_system_av')
plt.plot( time_v[1:-1], n_system[2:-1], label='n_system qj' ) 
os.chdir('..')

plt.legend()
plt.show()
fig.savefig('short_particle_current_n_tot_sites_' + str(n_tot)+ '.png')
