#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:03:50 2022

@author: reka
"""

import numpy as np 

import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice
#import evos.src.methods.lindblad_solver_reka as ed_mesoscopic_leads
import evos.src.methods.ed_time_dep_hamiltonian_lindblad_solver_new as solve
import matplotlib.pyplot as plt


def plot_expectation_values(results, ylabels=[], title=None, show_legend=False,
                            fig=None, axes=None, figsize=(8, 4)):

    if not isinstance(results, list):
        results = [results]

    n_e_ops = max([len(result.expect) for result in results])

    if not fig or not axes:
        if not figsize:
            figsize = (12, 3 * n_e_ops)
        fig, axes = plt.subplots(n_e_ops, 1, sharex=True,
                                 figsize=figsize, squeeze=False)

    for r_idx, result in enumerate(results):
        for e_idx, e in enumerate(result.expect):
            axes[e_idx, 0].plot(result.times, e,
                                label="%s [%d]" % (result.solver, e_idx))

    if title:
        axes[0, 0].set_title(title)

    axes[n_e_ops - 1, 0].set_xlabel("time", fontsize=12)
    for n in range(n_e_ops):
        if show_legend:
            axes[n, 0].legend()
        if ylabels:
            axes[n, 0].set_ylabel(ylabels[n], fontsize=12)

    return fig, axes

t_hop = 1
U = 1


alpha = 1

n_sites = 2
dim_H = 4 ** n_sites

T = 10
dt = 0.01
tsteps = int(T/dt)
t = np.linspace(0,T, tsteps)
#print(t)
#print(tsteps)


spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(n_sites)


def H(t): 
        
    hop = np.zeros((dim_H, dim_H), dtype = 'complex')
    for k in range(1, n_sites): 
        
        #hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k+1,N), c_down_dag(k,N))
        hop += np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down'))
     
 
    coul = np.zeros((dim_H, dim_H), dtype = 'complex')
    for k in range(1, n_sites+1): 
        n_up =  np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
        n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
          
        coul += np.dot(n_up,n_down)
    H = t_hop*hop + U*coul    
   
    return H

H1 = []
for t11 in t:
    H1.append(H(t11))
    
#print('h1 =', H1)

# alternate spin up down state: first site: up, second site: down, third site: up and so on... 
def state00_ket(n_sites):
            
    state_ket = np.zeros((dim_H, 1), dtype = 'complex')
    for i in range(0,dim_H+1):
        if i == 0:
            state_ket[i,0] = 1
    
    return state_ket


updown_ket = state00_ket(n_sites)

for i in np.arange(2,n_sites+1,2):
    #updown_ket = np.dot(c_up_dag(i-1, N), updown_ket)
    updown_ket = np.dot(spin_lat.sso('adag',i-1, 'up'), updown_ket)
    
    #updown_ket = np.dot(c_down_dag(i, N), updown_ket)
    updown_ket = np.dot(spin_lat.sso('adag',i, 'down'), updown_ket)



def L(k, N):
    n_up = np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up'))
    n_down = np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down'))
    
    L = alpha*(n_up + n_down) 
    return L


#print(L(2,3))

L_list_left = []
for k in range(0, n_sites):
    L_list_left.append(L(k+1, n_sites))

#print('exp =', H.dot(rho_updown).trace())
#print('exp_L =', L_list_left[1].dot(rho_updown).trace())
   
n_up_1 = np.dot(spin_lat.sso('adag',1, 'up'), spin_lat.sso('a',1, 'up'))
equation = solve.LindbladEquation(dim_H, H, L_list_left)
#lindblad = solve.TimeDepHamiltonianLindblad(dim_H, H, dt, T)
print(equation)
exp_n , t11  = solve.SolveLindbladEquation(dim_H, H, L_list_left, dt, T).solve(n_up_1, updown_ket)
#print(sol)
#print(lindblad.solve_lindblad_equation(updown_ket, H, L_list_left, n_up_1))



plt.plot(t11, exp_n, label='reka nfup')
plt.xlabel('t')
plt.ylabel('$< \hat n >$')
plt.legend()
plt.show()
