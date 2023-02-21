#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:15:07 2023

@author: reka
"""

import numpy as np 
import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice
import evos.src.methods.lindblad_mesoleads_solver as ed_mesoscopic_equation
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA

#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(threshold = False)

# Hamiltonian

t_hop = 0
U = 0
V = 0

eps = 1
kappa = 1
gamma = 1
alpha = 1

n_sites = 2 # number of system sites
n_lead_left = 1 # number of lindblad operators acting on leftest site
n_lead_right = 1 # number of lindblad operators acting on rightmost site

n_tot = n_sites + n_lead_left + n_lead_right

spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_tot)

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
T = 10
dt = 0.1
tsteps = int(T/dt)
t = np.linspace(0,T, tsteps)




def H_sys(t_hop, U, V): 
    
    # SYSTEM SITES 
    hop = np.zeros((dim_tot, dim_tot))
    for k in range(n_lead_left +1, n_tot - n_lead_right): 
        print('hopping terms on sites:', k, k+1)
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k+1,N), c_down_dag(k,N))
        hop += (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
     
 
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
    
    H = - t_hop*hop + U* coul + V*coul_nn 
    return H



def vac():
    state_ket = np.zeros((dim_tot, 1))
    for i in range(0,dim_tot+1):
        if i == 0:
            state_ket[i,0] = 1
    return state_ket

tot_init_state_ket = vac()
for i in range(1, n_tot+1): 
    if i <= n_lead_left: 
        tot_init_state_ket = 1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket)) #v2[index_low_lamb2].T#1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))

    if i > n_lead_left and i <= n_tot-n_lead_right:
        
        tot_init_state_ket =tot_init_state_ket # v[index_low_lamb].T# tot_init_state_ket
    
    if i > n_tot - n_lead_right:
        tot_init_state_ket = 1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket)) #v1[index_low_lamb1].T#1/(np.sqrt(2))* (np.dot(spin_lat.sso('adag',i, 'up'), tot_init_state_ket) + np.dot(spin_lat.sso('adag',i, 'down'), tot_init_state_ket))
        
tot_init_state_ket_norm = np.array(tot_init_state_ket/LA.norm(tot_init_state_ket), dtype = 'complex')




mes_leads = ed_mesoscopic_equation.MesoscopicLeads(n_tot, n_lead_left, n_lead_right, T_L, T_R, mu_L, mu_R, T, dt, eps_vector_l, eps_delta_vector_l, eps_vector_r, eps_delta_vector_r, k_vector_l, k_vector_r)

H_leads = mes_leads.H_leads_left() + mes_leads.H_leads_right()
H =  H_sys(t_hop,U, V) + H_leads

L_list = []#mes_leads.lindbladlistmesoscopic()
quit()
lindblad_equation = ed_mesoscopic_equation.LindbladEquation(n_tot, H, L_list).drho_dt

rho_sol = ed_mesoscopic_equation.SolveLindblad(n_tot).solve(tot_init_state_ket_norm, lindblad_equation, dt, T)


#trace preserved
#print(rho_sol[:,:,19].trace())

n_up_1 = np.dot(spin_lat.sso('adag',1, 'up'), spin_lat.sso('a',1, 'up'))
n_up_2 = np.dot(spin_lat.sso('adag',2, 'down'), spin_lat.sso('a',2, 'down'))
n_up_3 = np.dot(spin_lat.sso('adag',3, 'down'), spin_lat.sso('a',3, 'down'))
n_up_4 = np.dot(spin_lat.sso('adag',4, 'down'), spin_lat.sso('a',4, 'down'))

#exp = n_up_1.dot(rho_matrix).trace()
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
    

s = np.zeros((dim_tot, dim_tot), dtype = 'complex')
# expectation value of current of down spins through wire
for k in range(n_lead_left + 1, n_tot - n_lead_right +1):
    print('k=', k)
    #s += -1j * ( np.dot(spin_lat.sso('adag', k, 'up'), spin_lat.sso('a', k, 'down') ) * np.dot(spin_lat.sso('adag', k+1, 'down'),spin_lat.sso('a', k+1, 'up') ) - np.dot(spin_lat.sso('adag', k+1, 'up'),spin_lat.sso('a', k+1, 'down') ) * np.dot(spin_lat.sso('adag', k, 'down'),spin_lat.sso('a', k, 'up') ) )
    s += -1j* ( np.dot(np.dot(spin_lat.sso('adag', k, 'up'), spin_lat.sso('a', k, 'down') ) , np.dot(spin_lat.sso('adag', k+1, 'down'),spin_lat.sso('a', k+1, 'up') )) ) #- np.dot( np.dot(spin_lat.sso('adag', k+1, 'up'),spin_lat.sso('a', k+1, 'down') ) , np.dot(spin_lat.sso('adag', k, 'down'),spin_lat.sso('a', k, 'up') ) ))
print('s', np.where(s!=0))


exp_s = []
t1 = []
for i in range(0, tsteps):
    exp = s.dot(rho_sol[:,:,i]).trace()
    exp_s.append(exp.imag)
    t1.append(i)


j_left = -1j*( np.dot(spin_lat.sso('adag',n_lead_left, 'down'), spin_lat.sso('a',n_lead_left +1, 'down')) - np.dot(spin_lat.sso('adag',n_lead_left+1, 'down'), spin_lat.sso('a',n_lead_left, 'down')))
j_right = -1j*( np.dot(spin_lat.sso('adag',n_lead_left + n_sites, 'down'), spin_lat.sso('a',n_lead_left + n_sites + 1, 'down')) - np.dot(spin_lat.sso('adag',n_lead_left + n_sites +1, 'down'), spin_lat.sso('a',n_lead_left + n_sites, 'down')))

j = j_left + j_right


exp_j = []
t1 = []
for i in range(0, tsteps):
    exp = j.dot(rho_sol[:,:,i]).trace()
    exp_j.append(exp.imag)
    t1.append(i)
    
    
exp_j1 = []
for i in range(0, tsteps):
    exp1 = j.dot(rho_sol[:,:,i]).trace()
    exp_j1.append(exp1.real)
    t1.append(i)
    
opt_cond = []
for i in range(0, tsteps):
    opt_conductivity = (exp_j[i]+exp_j1[i])/(mu_L - mu_R)
    opt_cond.append(opt_conductivity)
   
beta_L = np.exp(- 1/T_L * (eps_vector_l[0] - mu_L) ) / ( np.exp(- 1/T_L * (eps_vector_l[0]-mu_L) ) + 1)

beta_list_L = []
for i in range(len(t)):
    beta_list_L.append(beta_L)
    
beta_R = np.exp( - 1/T_R * (eps_vector_r[0] - mu_R) ) / ( np.exp(- 1/T_R * (eps_vector_r[0]-mu_R) ) + 1)

beta_list_R = []
for i in range(len(t)):
    beta_list_R.append(beta_R)
    
limit_1 = []
for i in range(len(t)):
    limit_1.append(0.09)
    
limit_2 = []
for i in range(len(t)):
    limit_2.append(0.0)

#plt.plot(t, beta_list_L, label = 'analytic thermalized expect. val', linestyle = 'dashed')    
#plt.plot(t, beta_list_R, label = 'analytic thermalized expect. val', linestyle = 'dashed')    
#plt.plot(t, limit_1, label = "0.09", linestyle = 'dashed')
#plt.plot(t, limit_2, label = "0.0", linestyle = 'dashed')
#print(N_up(1,N))
 
plt.plot(t, exp_n_up_lead_left, label='$< \hat n > $ down spins on left lead')
#plt.plot(t, exp_n_up_first_sys_site, label='$< \hat n >$ down spins on first sys site')
#plt.plot(t, exp_n_up_second_sys_site, label='$< \hat n >$ down spins on second sys site')
#plt.plot(t, exp_n_up_lead_right, label='$< \hat n >$ down spins on right lead')


#plt.plot(t, exp_j, label='$< \hat j > $ imag' )
#plt.plot(t, exp_j1, label='$< \hat j > $ real')
#plt.plot(t, opt_cond, label = '$\sigma$')

plt.xlabel('t')
plt.ylabel('$< \hat j >, < \hat n >$, $\sigma$')
plt.title('thermalization of extended Hubbard chain')
#plt.title('$L_{1} = 2\hat a_{down, dag,2}, L_{2} =  \hat a_{down,2}$')
#plt.savefig('FH_2sites_4p_6.pdf')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

'''
##################################################################################################################################
# FIT OPTICAL CONDUCTIVITY  

np.savetxt('spin_current', exp_s)

np.savetxt('time', t)

'''
