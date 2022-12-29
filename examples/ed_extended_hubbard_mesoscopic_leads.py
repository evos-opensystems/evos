#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:32:13 2022

@author: reka
"""

import numpy as np 
from scipy.integrate import solve_ivp
#import spinful_fermions_lattice as spinful_fermions_lattice
import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice
import evos.src.methods.lindblad_solver_reka as ed_mesoscopic_leads
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as LA
import sys
import math



#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(threshold = False)

# Hamiltonian
J = 1
U = 1
V = 1
eps = 1
kappa = 1

alpha = 1

n_sites = 2 # number of system sites
n_lead_left = 1 # number of lindblad operators acting on leftest site
n_lead_right = 1 # number of lindblad operators acting on rightmost site

n_tot = n_sites + n_lead_left + n_lead_right

dim_H_sys = 4 ** n_sites

dim_H_lead_left =  4** n_lead_left 
dim_H_lead_right =  4** n_lead_right

dim_tot = dim_H_sys*dim_H_lead_left*dim_H_lead_right


# temperature and chemical potential on the different leads
T_L = 1
T_R = 1
mu_L = 1
mu_R = -1

####################################################################################################################
# spectral function
def const_spec_funct(G,W,eps):
    if eps >= -W and eps <= W:
        return G
    else:
        return 0

G = 1
W= 1
#eps_step = 2*W/n_lead_left

#LEFT LEAD COEEFICIENTS
eps_step_l = 2* W / n_lead_left 
eps_vector_l = np.arange( -W, W, eps_step_l )
eps_delta_vector_l = eps_step_l * np.ones( len(eps_vector_l) )

k_vector_l = np.zeros( len(eps_vector_l) )
for i in range( len(eps_vector_l) ):
    k_vector_l[i] = np.sqrt( const_spec_funct( G , W, eps_vector_l[i] ) * eps_delta_vector_l[i]/ (2*math.pi) )  
    
#RIGHT LEAD COEEFICIENTS
eps_step_r = 2* W / n_lead_right
eps_vector_r = np.arange( -W, W, eps_step_r )
eps_delta_vector_r = eps_step_r * np.ones( len(eps_vector_r) )

k_vector_r = np.zeros( len(eps_vector_r) )
for i in range( len(eps_vector_r) ):
    k_vector_r[i] = np.sqrt( const_spec_funct( G , W, eps_vector_r[i] ) * eps_delta_vector_r[i]/ (2*math.pi) ) 
    
########################################################################################################################

# paramters (time, ...) for solving differential equation
T = 40
dt = 0.5
tsteps = int(T/dt)
t = np.linspace(0,T, tsteps)
#print(tsteps)


def fermi_dist(beta, e, mu):
    f = 1 / ( np.exp( beta * (e-mu) ) + 1)
    return f

spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_tot)
#sys_spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_sites)


def H_sys(J, U, V): 
    
    # SYSTEM SITES 
    hop = np.zeros((dim_tot, dim_tot))
    for k in range(n_lead_left +1, n_tot - n_lead_right): 
        print('hopping terms on sites:', k, k+1)
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k+1,N), c_down_dag(k,N))
        hop += np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down'))
     
 
    coul = np.zeros((dim_tot, dim_tot))
    for k in range(n_lead_left +1, n_tot - n_lead_right +1): 
        print('coulomb terms on sites:', k)
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
          
        coul += np.dot(n_up,n_down)
        
    
    
    coul_nn = np.zeros((dim_tot, dim_tot))
    for k in range(n_lead_left +1, n_tot - n_lead_right):
        print('coulomb neighbour terms on sites:', k, k+1)
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
        
        n_up_nn = np.dot(spin_lat.sso('adag',k+1, 'up'), spin_lat.sso('a',k+1, 'up'))
        n_down_nn = np.dot(spin_lat.sso('adag',k+1, 'down'), spin_lat.sso('a',k+1, 'down'))
        
        
        coul_nn += np.dot(n_up, n_up_nn) + np.dot(n_up, n_down_nn) + np.dot(n_down, n_up_nn) + np.dot(n_down, n_down_nn)
    
    H = - J*hop + U* coul + V*coul_nn 
    return H


def H_leads_left(eps, k_vec, mu_L):
    # LEAD SITES - kinetic energy of left leads
    kin_leads = np.zeros((dim_tot, dim_tot))
    if n_lead_left == 0: 
        print('Ekin_lead left terms on sites:', 0)
        for k in range(0, dim_tot):
            kin_leads = np.zeros((dim_tot, dim_tot))
    else: 
        for k in range(1, n_lead_left+1): 
            print('Ekin_lead left terms on sites:', k)
            kin_leads += (eps[k-1]) *( np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up')) + np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down')))
     
    
    
    # HOPPING BETWEEN LEADS AND SYSTEM LEFT SIDE
    hop_sys_lead = np.zeros((dim_tot, dim_tot))
    if n_lead_left == 0: 
        print('left sys lead hopping on sites:', 0)
        for k in range(0, dim_tot): 
            hop_sys_lead = np.zeros((dim_tot, dim_tot))
    else: 
        for k in range(n_lead_left, n_lead_left+1): 
            print('left sys lead hopping on sites:', k, k+1)
            hop_sys_lead += k_vec[k-1]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
     
        
    H = kin_leads + hop_sys_lead    
    return H

def H_leads_right(eps,k_vec, mu_R):
     # LEAD SITES - kinetic energy of left leads
    kin_leads = np.zeros((dim_tot, dim_tot))
    if n_lead_right == 0: 
        print('Ekin_lead right terms on sites:', 0)
        for k in range(0, dim_tot): 
            kin_leads = np.zeros((dim_tot, dim_tot))
    else:       
        for k in range(n_tot - n_lead_right + 1, n_tot+1):    
            print('Ekin_lead right terms on sites:', k)
            kin_leads += (eps[k - (n_tot - n_lead_right +1)] ) *( np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up')) + np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down')))
     
    # HOPPING BETWEEN LEADS AND SYSTEM RIGHT SIDE
    hop_sys_lead = np.zeros((dim_tot, dim_tot))
    if n_lead_right == 0: 
        print('right sys lead hopping on sites:', 0)
        for k in range(0, dim_tot):
            hop_sys_lead = np.zeros((dim_tot, dim_tot))
    else: 
        for k in range(n_tot - n_lead_right , n_tot): 
            print('right sys lead hopping on sites:', k, k+1)
            hop_sys_lead += k_vec[k - (n_tot - n_lead_right) ]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
    H = kin_leads + hop_sys_lead       
    return H

H = H_sys(U,J, V) + H_leads_left(eps_vector_l, k_vector_l, mu_L) + H_leads_right(eps_vector_r, k_vector_r, mu_R)

#print(H)

###################################################################################################################################
# PREPARE LEADS IN THERMAL STATE
'''
rho_leads_left = np.zeros((dim_H_lead_left) , dtype = 'complex')
rho_leads_left = expm( 1/T_L* H_leads_left(eps,kappa))/np.trace(expm(1/T_L * H_leads_left(eps, kappa)))

print(np.trace(rho_leads_left))

rho_leads_right = np.zeros((dim_H_lead_right) , dtype = 'complex')
rho_leads_right = expm( 1/T_L* H_leads_right(eps,kappa))/np.trace(expm(1/T_L * H_leads_right(eps, kappa)))

print(np.trace(rho_leads_right))
'''
# PREPARE SYSTEM SITES IN ANY CONVENIENT STATE

# no occupation of sites:
def vac():
            
    state_ket = np.zeros((dim_tot, 1))
    for i in range(0,dim_tot+1):
        if i == 0:
            state_ket[i,0] = 1

    return state_ket


#print(dim_tot)



tot_init_state_ket = vac()
for i in range(1, n_tot+1): 
    if i <= n_lead_left: 
        tot_init_state_ket = 1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))

    if i > n_lead_left and i <= n_tot-n_lead_right:
        tot_init_state_ket = tot_init_state_ket
    
    if i > n_tot - n_lead_right:
        tot_init_state_ket = 1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))

        



tot_init_state_ket_norm = tot_init_state_ket/LA.norm(tot_init_state_ket)
# total density matrix: 

tot_init_state_bra = np.conjugate(tot_init_state_ket_norm) 
rho_updown = np.outer(tot_init_state_ket_norm, tot_init_state_bra)   
#print('sum =', LA.norm(tot_init_state_ket_norm))
 


rho_matrix = rho_updown
#print('trace of total density matrix = ', np.trace(rho_matrix))
#print(rho_matrix)
rho_vec = []
for i in range(0, dim_tot):
    for  j in range(0 ,dim_tot):
        rho_vec.append(rho_matrix[i,j])        
rho_vec = np.array(rho_vec,dtype='complex')




#print(L(2,3))

L_list = []

for k in range(1, n_lead_left+1):
    print('k_left = ', k)
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* np.exp( 1/T_L * (eps_vector_l[k-1] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[k-1], mu_L))* spin_lat.sso('a',k, 'up'))
    #print(spin_lat.sso('a', k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* np.exp( 1/T_L * (eps_vector_l[k-1] - mu_L) ) * fermi_dist(1/T_L, eps_vector_l[k-1], mu_L)) * spin_lat.sso('a',k, 'down'))
    
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* fermi_dist(1/T_L, eps_vector_l[k-1], mu_L)) * spin_lat.sso('adag',k, 'up'))
    #print(spin_lat.sso('a', k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_l[k-1]* fermi_dist(1/T_L, eps_vector_l[k-1], mu_L)) * spin_lat.sso('adag',k, 'down'))
    
  
    
for k in range(n_tot-n_lead_right +1, n_tot+1):
    print('k_right = ', k)
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* np.exp( 1/T_R * (eps_vector_r[k-(n_tot-n_lead_right +1)] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R))* spin_lat.sso('a',k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* np.exp( 1/T_R * (eps_vector_r[k-(n_tot-n_lead_right +1)] - mu_R) ) * fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R))* spin_lat.sso('a',k, 'down'))
    
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R)) * spin_lat.sso('adag',k, 'up'))
    L_list.append( np.sqrt( eps_delta_vector_r[k-(n_tot-n_lead_right +1)]* fermi_dist(1/T_R, eps_vector_r[k-(n_tot-n_lead_right +1)], mu_R)) * spin_lat.sso('adag',k, 'down'))
    


#L_list = []
#for k in range(0, n_sites):
    
dyn = ed_mesoscopic_leads.MesoscopicLeadsLindblad(dim_tot, H, L_list)


sol = solve_ivp(dyn.drho_dt, (0,T), rho_vec, t_eval=t)        
#print(sol.y)


#plot some expectation value at each time step
#time dependant rho:
rho_sol = np.zeros((dim_tot,dim_tot, tsteps),dtype='complex')
count=0
for n in range(dim_tot):
    for  m in range(0,dim_tot):
        rho_sol[n,m,:] = sol.y[count,:]
        count+=1
    
for n in range(dim_tot):
    for  m in range(0,dim_tot):
        rho_sol[n,m,:] = np.conjugate(rho_sol[m,n])

#trace preserved
#print(rho_sol[:,:,19].trace())

n_up_1 = np.dot(spin_lat.sso('adag',1, 'down'), spin_lat.sso('a',1, 'down'))
n_up_2 = np.dot(spin_lat.sso('adag',2, 'down'), spin_lat.sso('a',2, 'down'))
n_up_3 = np.dot(spin_lat.sso('adag',3, 'down'), spin_lat.sso('a',3, 'down'))
n_up_4 = np.dot(spin_lat.sso('adag',4, 'down'), spin_lat.sso('a',4, 'down'))

exp = n_up_1.dot(rho_matrix).trace()
#print('exp = ', exp)

#compute expectation value
exp_n_up_lead_left = []
t1 = []
for i in range(0, tsteps):
    exp = n_up_1.dot(rho_sol[:,:,i]).trace()
    exp_n_up_lead_left.append(exp)
    t1.append(i)
    
exp_n_up_first_sys_site = []
t1 = []
for i in range(0, tsteps):
    exp = n_up_2.dot(rho_sol[:,:,i]).trace()
    exp_n_up_first_sys_site.append(exp)
    t1.append(i)
    
    
exp_n_up_second_sys_site = []
t1 = []
for i in range(0, tsteps):
    exp = n_up_3.dot(rho_sol[:,:,i]).trace()
    exp_n_up_second_sys_site.append(exp)
    t1.append(i)
    
    
exp_n_up_lead_right = []
t1 = []
for i in range(0, tsteps):
    exp = n_up_4.dot(rho_sol[:,:,i]).trace()
    exp_n_up_lead_right.append(exp)
    t1.append(i)
    


# expectation value of current of down spins through wire

j_left = np.dot(spin_lat.sso('adag',n_lead_left, 'down'), spin_lat.sso('a',n_lead_left +1, 'down')) + np.dot(spin_lat.sso('adag',n_lead_left+1, 'down'), spin_lat.sso('a',n_lead_left, 'down'))
j_right = np.dot(spin_lat.sso('adag',n_lead_left + n_sites, 'down'), spin_lat.sso('a',n_lead_left + n_sites + 1, 'down')) + np.dot(spin_lat.sso('adag',n_lead_left + n_sites +1, 'down'), spin_lat.sso('a',n_lead_left + n_sites, 'down'))

j = j_left#+ j_right

exp_j = []
t1 = []
for i in range(0, tsteps):
    exp = j.dot(rho_sol[:,:,i]).trace()
    exp_j.append(exp)
    t1.append(i)
    
beta_L = np.exp(- 1/T_L * (eps_vector_l[0] - mu_L) ) / ( np.exp(- 1/T_L * (eps_vector_l[0]-mu_L) ) + 1)

beta_list_L = []
for i in range(len(t)):
    beta_list_L.append(beta_L)
    
beta_R = np.exp( - 1/T_R * (eps_vector_r[0] - mu_R) ) / ( np.exp(- 1/T_R * (eps_vector_r[0]-mu_R) ) + 1)

beta_list_R = []
for i in range(len(t)):
    beta_list_R.append(beta_R)
    
    
plt.plot(t, beta_list_L, label = 'analytic thermalized expect. val', linestyle = 'dashed')    
plt.plot(t, beta_list_R, label = 'analytic thermalized expect. val', linestyle = 'dashed')    
 
#print(N_up(1,N))
 
plt.plot(t, exp_n_up_lead_left, label='< \hat n > down spins on left lead')
plt.plot(t, exp_n_up_first_sys_site, label='< \hat n > down spins on first sys site')
plt.plot(t, exp_n_up_second_sys_site, label='< \hat n > down spins on second sys site')
plt.plot(t, exp_n_up_lead_right, label='number of down spins on right lead')


plt.plot(t, exp_j, label='$< \hat j >$')

plt.xlabel('t')
plt.ylabel('$< \hat j >, < \hat n >$')
#plt.title('$L_{1} = 2\hat a_{down, dag,2}, L_{2} =  \hat a_{down,2}$')
#plt.savefig('FH_2sites_4p_6.pdf')

plt.legend()

plt.show()






















