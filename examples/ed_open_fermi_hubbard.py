#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:03:50 2022

@author: reka
"""

import numpy as np 
from scipy.integrate import solve_ivp
import spinful_fermions_lattice as spinful_fermions_lattice
#import ed_lindblad_class as ed_mesoscopic_leads
import evos.src.methods.lindblad_solver_reka as ed_mesoscopic_leads
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


J = 1
U = 1

T = 1
dt =1
alpha = 1

n_sites = 2
dim_H = 4 ** n_sites

T = 10
dt = 0.1
tsteps = int(T/dt)
t = np.linspace(0,T, tsteps)
print(tsteps)


spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_sites)


def H(J, U): 
        
    hop = np.zeros((dim_H, dim_H))
    for k in range(1, n_sites): 
        
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k+1,N), c_down_dag(k,N))
        hop += np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down'))
     
 
    coul = np.zeros((dim_H, dim_H))
    for k in range(1, n_sites+1): 
        n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
          
        coul += np.dot(n_up,n_down)
        
    
        
    H = - J* hop + U* coul
    return H

H = H(U,J)


def state00_ket(n_sites):
    
    state_bra = np.zeros((1, dim_H))
    for i in range(0,dim_H+1):
        
        if i == 0:
            state_bra[0,i] = 1
            
    state_ket = np.zeros((dim_H, 1))
    for i in range(0,dim_H+1):
        if i == 0:
            state_ket[i,0] = 1
            
    state_bra = np.conjugate(state_ket)
    
    return state_ket


updown_ket = state00_ket(n_sites)
for i in np.arange(2,n_sites+1,2):
    #updown_ket = np.dot(c_up_dag(i-1, N), updown_ket)
    updown_ket = np.dot(spin_lat.sso('adag',i-1, 'up'), updown_ket)
    
    #updown_ket = np.dot(c_down_dag(i, N), updown_ket)
    updown_ket = np.dot(spin_lat.sso('adag',i, 'down'), updown_ket)

updown_bra = np.conjugate(updown_ket) 
   
rho_updown = np.outer(updown_ket, updown_bra)     

rho_matrix = rho_updown
#print(rho_matrix)
rho_vec = []
for i in range(0, dim_H):
    for  j in range(0 ,dim_H):
        rho_vec.append(rho_matrix[i,j])        
rho_vec = np.array(rho_vec,dtype='complex')



def L(k, N):
    n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
    n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
    
    L = alpha*(n_up + n_down) 
    return L


#print(L(2,3))

L_list_left = []
for k in range(0, n_sites):
    L_list_left.append(L(k+1, n_sites))


   
n_up_1 = np.dot(spin_lat.sso('adag',1, 'up'), spin_lat.sso('a',1, 'up'))


    
dyn = ed_mesoscopic_leads.MesoscopicLeadsLindblad(dim_H, H, L_list_left)

sol = solve_ivp(dyn.drho_dt, (0,T), rho_vec, t_eval=t)        
#print(sol.y)


#plot some expectation value at each time step
#time dependant rho:
rho_sol = np.zeros((dim_H,dim_H, tsteps),dtype='complex')
count=0
for n in range(dim_H):
    for  m in range(0,dim_H):
        rho_sol[n,m,:] = sol.y[count,:]
        count+=1
    
for n in range(dim_H):
    for  m in range(0,dim_H):
        rho_sol[n,m,:] = np.conjugate(rho_sol[m,n])

#trace preserved
#print(rho_sol[:,:,19].trace())


#compute expectation value
exp_n = []
t1 = []
for i in range(0, tsteps):
    exp = n_up_1.dot(rho_sol[:,:,i]).trace()
    exp_n.append(exp)
    t1.append(i)
    
 
#print(N_up(1,N))
    
plt.plot(t, exp_n, label='reka nfup')
plt.xlabel('t')
plt.ylabel('$< \hat n >$')
#plt.title('$L_{1} = 2\hat a_{down, dag,2}, L_{2} =  \hat a_{down,2}$')
#plt.savefig('FH_2sites_4p_6.pdf')

plt.legend()

