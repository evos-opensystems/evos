#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:22:38 2022

@author: reka
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as LA
from numpy import random
#import sobol_seq
from scipy.stats import qmc


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 11})

############################################################################################
N = 2 #number of sites

J = 1
#k = 1
t = 1
U = 1

dim = 4**N

alpha = 1
#Quantum jump solver parameters:

delta_t = 0.01
dt = delta_t
N1 = 100

#number of repititions:
repititions = 10

sampler = qmc.Sobol(d=2, scramble=False)
sample = sampler.random_base2(m=20)

ra = sample[:,0]
ra2 = sample[:,1]
#print(sample)
#print(r)
############################################################################################

def delta(k,l):
    return 1 if k == l else 0

#operators used later 
iden = np.array([[ 1,0],[ 0,1]])
P = np.array([[ 1,0],[ 0,-1]]) #parity
a_2 = np.array([[ 0,1],[ 0,0]]) #annihilation operator 
adag_2 = np.array([[ 0,0],[ 1,0]]) #creation operator
aadag = np.kron(a_2, adag_2)
#print(aadag)
adaga= np.kron(adag_2, a_2)
n_in = np.dot(adag_2, a_2)
#print(n_in)

# operators on one single site, which can have spin up & spin down
a_up = np.kron(a_2, P) #annihilate spin up
a_down = np.kron( iden, np.dot(a_2, iden)) #annihilate spin down
a_down_dag = np.kron(iden, np.dot(adag_2, iden)) # create spin down
a_up_dag = np.kron( adag_2, P) # annihilate spin up



P_r1 = np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
P_r = np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])

#ONE SINGLE SITE WITHOUT THE REST
#for the hopping terms: 
a_up_1 = np.kron(np.dot(a_2,P), P)
a_up_dag_1 = np.kron(np.dot(adag_2,P), P)
a_down_1 = np.kron(P, np.dot(P, a_2))
a_down_dag_1 = np.kron(P, np.dot(P, adag_2))
#print(a_down_1)

c_up_r1 = np.kron(a_up_1, a_up_dag)
c_uP_r1 = np.kron(a_up_dag_1, a_up)
c_down_r1 = np.kron(a_down, a_down_dag_1)
c_down_l1 = np.kron(a_down_dag, a_down_1)

#for the coulomb terms
n_up = np.kron(np.dot(adag_2, a_2), iden)
n_down = np.kron(iden, np.dot(adag_2, a_2))
n_updown = np.kron(n_in, iden) + np.kron(iden, n_in)

#print(np.dot(adag_2, a_2))
#print(n_updown)


##################################################################################################
#different operators
#LOCAL TERMS
#create/annihilate spin up/down on one single site, in whole hilbert space with dimension 4**N

def c_up_dag(k, N):
    #identities to the left
    iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                iden_l[i,j] = 1
    #print(iden_l)  
    #print(a_up_dag)          
    i = N-k
        #print(i)
    if i == 0:
            P_r1 = 1
    if i == 1:
        #print(i)
        P_r1 = P_r
    #parity operators to the right
    if i > 1: 
        #print(i)
        P_r1 = P_r
        for l in range(0,i-1):
            #print(l)
            P_r1 = np.kron(P_r1, P_r*(l+1)/(l+1))
            #print(P_r1, np(P_r1))
    #print(P_r1)
    #a = np.kron(P_r1, np.kron(a_up_dag, iden_l))
    a = np.kron(iden_l, np.kron(a_up_dag, P_r1))
    return a

def c_down_dag(k, N):
    #identities to the left
    iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                iden_l[i,j] = 1
    #print(iden_l)  
    #print(a_up_dag)          
    i = N-k
        #print(i)
    if i == 0:
            P_r1 = 1
    if i == 1:
        #print(i)
        P_r1 = P_r
    #parity operators to the right
    if i > 1: 
        #print(i)
        P_r1 = P_r
        for l in range(0,i-1):
            #print(l)
            P_r1 = np.kron(P_r1, P_r*(l+1)/(l+1))
            #print(P_r1, np(P_r1))
    #print(P_r1)
    #a = np.kron(P_r1, np.kron(a_up_dag, iden_l))
    a = np.kron(iden_l, np.kron(a_down_dag, P_r1))
    return a

def c_up(k, N):
     #identities to the left
    iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                iden_l[i,j] = 1
    #print(iden_l)  
    #print(a_up_dag)          
    i = N-k
        #print(i)
    if i == 0:
            P_r1 = 1
    if i == 1:
        #print(i)
        P_r1 = P_r
    #parity operators to the right
    if i > 1: 
        #print(i)
        P_r1 = P_r
        for l in range(0,i-1):
            #print(l)
            P_r1 = np.kron(P_r1, P_r*(l+1)/(l+1))
            #print(P_r1, np(P_r1))
    #print(P_r1)
    #a = np.kron(P_r1, np.kron(a_up_dag, iden_l))
    a = np.kron(iden_l, np.kron(a_up, P_r1))
    return a

def c_down(k, N):
    #identities to the left
    iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                iden_l[i,j] = 1
    #print(iden_l)  
    #print(a_up_dag)          
    i = N-k
        #print(i)
    if i == 0:
            P_r1 = 1
    if i == 1:
        #print(i)
        P_r1 = P_r
    #parity operators to the right
    if i > 1: 
        #print(i)
        P_r1 = P_r
        for l in range(0,i-1):
            #print(l)
            P_r1 = np.kron(P_r1, P_r*(l+1)/(l+1))
            #print(P_r1, np(P_r1))
    #print(P_r1)
    #a = np.kron(P_r1, np.kron(a_up_dag, iden_l))
    a = np.kron(iden_l, np.kron(a_down, P_r1))
    return a


#identity
def idty(N):
    identity = np.zeros((4**N, 4**N))
    for i in range(0, 4**N): 
        identity[i,i] = 1
        
    return identity
        

#NUMBER OPERATORS
#UP + DOWN
def n(k, N):
    iden_l = np.zeros((4**(N-k), 4**(N-k)))
    for i in range(0, 4**(N-k)):
        for j in range(0, 4**(N-k)):
            if i == j:
                iden_l[i,j] = 1
    
    Iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                Iden_l[i,j] = 1
           
    
    a = np.kron(Iden_l, np.kron(n_updown, iden_l))
    
    return a
    
#print(n(1, 2) + n(2,2))   
# ONLY UP
def N_up(k, N):
    iden_l = np.zeros((4**(N-k), 4**(N-k)))
    for i in range(0, 4**(N-k)):
        for j in range(0, 4**(N-k)):
            if i == j:
                iden_l[i,j] = 1
    
    Iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                Iden_l[i,j] = 1
           
    
    a = np.kron(Iden_l, np.kron(n_up, iden_l))
    
    return a

#ONLY DOWN
def N_down(k, N):
    iden_l = np.zeros((4**(N-k), 4**(N-k)))
    for i in range(0, 4**(N-k)):
        for j in range(0, 4**(N-k)):
            if i == j:
                iden_l[i,j] = 1
    
    Iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                Iden_l[i,j] = 1
           
    
    a = np.kron(Iden_l, np.kron(n_down, iden_l))
    
    return a

#COULOMB TERM NUMBER OPERATOR
def n_up_down(k, N):
    iden_l = np.zeros((4**(N-k), 4**(N-k)))
    for i in range(0, 4**(N-k)):
        for j in range(0, 4**(N-k)):
            if i == j:
                iden_l[i,j] = 1
    
    Iden_l = np.zeros((4**(k-1), 4**(k-1)))
    for i in range(0, 4**(k-1)):
        for j in range(0, 4**(k-1)):
            if i == j:
                Iden_l[i,j] = 1
           
    
    a = np.kron(Iden_l, np.kron(np.dot(n_up, n_down), iden_l))
    
    return a
    
    
    
#Hamiltonian
# Couldomb repulsion on site
# hopping between sites

def H(J, U, N): 
        
    hop = np.zeros((dim, dim), dtype = complex)
    for k in range(1, N): 
        
        hop += np.dot(c_up(k,N), c_up_dag(k + 1 ,N)) + np.dot(c_up(k + 1,N), c_up_dag(k,N)) + np.dot(c_down(k,N), c_down_dag(k +1 ,N)) + np.dot(c_down(k +1,N), c_down_dag(k,N))
        
    Coul = np.zeros((dim, dim), dtype = complex)
    for k in range(1, N+1): 
        #print(Sx_nn_new)
        Coul += n_up_down(k, N)
        
    
        
    H = - J* hop + U* Coul
    return H

H = H(J, U, N)

#Lindblad operators for each site in a list
def L(k, N):
    L = alpha*(N_up(k, N) + N_down(k, N)) 
    return L
#print(L(2,3))

L_list = []
for k in range(0,N):
    L_list.append(L(k+1,N))
    
#construct effective Hamiltonian
L_sum = np.zeros((dim,dim), dtype = complex)
for k in range(0, N):
    L_sum = L_sum - 1j/2*L_list[k].T.conj().dot(L_list[k])

Heff = H + L_sum 


#state
#vacuum state
def state00_ket(N):
    
    state_bra = np.zeros((1, dim))
    for i in range(0,dim+1):
        
        if i == 0:
            state_bra[0,i] = 1
            
    state_ket = np.zeros((dim, 1))
    for i in range(0,dim+1):
        if i == 0:
            state_ket[i,0] = 1
            
    state_bra = np.conjugate(state_ket)
    rho = np.outer(state_ket, state_bra)     
    #print(state_bra)
    return state_ket



#state with alternate up down fermions
updown_ket = state00_ket(N)   
for i in np.arange(2,N+1,2):
    updown_ket = np.dot(c_up_dag(i-1, N), updown_ket)
    updown_ket = np.dot(c_down_dag(i, N), updown_ket)
        
updown_bra = np.conjugate(updown_ket) 
   
rho_updown = np.outer(updown_ket, updown_bra)


######################################################################################
#run program 
def Select(L_l, phi):
    interval = np.zeros(N+1) # as many lindblad operators as sites
    interval[0] = 0
    
    for i in range(0,N):
        
        phi1 = L_list[i].dot(phi)
        norm = LA.norm(phi1)
        interval[i+1] = interval[i] + norm**2
        
    norm_int = interval/interval[N]
    #print(norm_int)
    #r2 = np.random.uniform(low = 0.0, high = 1.0, size = None) 
    r2 = random.rand()
    #r2 =ra2[i]
    
    for i in range(0,N):
        if r2>= norm_int[i] and r2<= norm_int[i+1]:
            
            sel = L_list[i].dot(phi)
            break
    return sel
            
      

def QJMC(Heff, phi0, i):
    phi_new = phi0
    
    
    T = []
    Mean_n_t = []
    
    U = expm(-1j*Heff*dt)
    
    for n in range(1,N1):
        
        
        T.append(delta_t*n)
        
        
        #newrho_ket = (idty(N)-1j*Heff*dt).dot(phi_new)
        newrho_ket = U.dot(phi_new)
        norm_2 = LA.norm(newrho_ket)
        #print(norm)
        dp = 1- norm_2**2
        #print(dp)
        
        #r = np.random.uniform(low = 0.0, high = 1.0, size = None) 
        r = random.rand()
        #r = ra[n]
        
        if r > dp:
            phi_new = newrho_ket/(np.sqrt(1-dp))
        
        
        elif r <= dp:
            newrho_ket1 = Select(L_list, phi_new)
            norm1 = LA.norm(newrho_ket1)
            #dp1 = np.abs(1-norm1_2**2)
            phi_new = newrho_ket1/norm1
        #compute some observable 
        rho = np.kron(phi_new.T.conj(), phi_new)
        #print(rho)
        mean_n_t = N_up(1,N).dot(rho).trace()
        Mean_n_t.append(mean_n_t)
        Mean_n_t1 = np.array(Mean_n_t)
    return Mean_n_t1, T



mean_n, T = QJMC(Heff, updown_ket, i)
#quantum jump one time repeatedly for averaging later
Mean_n = np.zeros(((1), (len(mean_n))))


for i in range (0, repititions): 
    mean_n, T = QJMC(Heff, updown_ket, i)
    #print(meanx)
    for k in range(0, len(mean_n)):
        Mean_n[:,k] = Mean_n[:,k] + mean_n[k]
    #plt.plot(T, mean)
    #
MEAN_N = Mean_n/(repititions+1)
#print(Meanx/repititions)    
#print(np.size(np.array(Meanx.T)))
#print(len(T))



plt.plot(T, MEAN_N.T, label='reka QJ, dt = 0.01')
plt.title('1000 repititions')
plt.legend()
