
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

import numpy as np
from scipy.linalg import expm
from numpy import linalg as LA
from numpy import random



class Solve():
    
    def heff(H, L_list): 
        #construct effective Hamiltonian
        L_sum = np.zeros((len(H),len(H)), dtype = complex)
        for k in range(0, len(L_list)):
            L_sum = L_sum - 1j/2*L_list[k].T.conj().dot(L_list[k])
        
        Heff = H + L_sum
        
        return Heff
    
    
    def Select(L_list, phi, len_L_list):
        interval = np.zeros(len_L_list) # as many lindblad operators as sites
        interval[0] = 0
        
        for i in range(0,len_L_list):
            
            phi1 = L_list[i].dot(phi)
            norm = LA.norm(phi1)
            interval[i+1] = interval[i] + norm**2
            
        norm_int = interval/interval[len_L_list]
        #print(norm_int)
        #r2 = np.random.uniform(low = 0.0, high = 1.0, size = None) 
        r2 = random.rand()
        #r2 =ra2[i]
        
        for i in range(0,len_L_list):
            if r2>= norm_int[i] and r2<= norm_int[i+1]:
                
                sel = L_list[i].dot(phi)
                break
        return sel
    
    def QJMC(cls, Heff, phi0, L_list, observable, i, delta_t, N1):
        
        len_L_list = len(L_list)
        
        dt = delta_t
        
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
            #print('r = ', r)
            #r = ra[n]
            
            if r > dp:
                phi_new = newrho_ket/(np.sqrt(1-dp))
            
            
            elif r <= dp:
                newrho_ket1 = cls.Select(L_list, phi_new,len_L_list)
                 
                norm1 = LA.norm(newrho_ket1)
                #dp1 = np.abs(1-norm1_2**2)
                phi_new = newrho_ket1/norm1
            #compute some observable 
            rho = np.kron(phi_new.T.conj(), phi_new)
            #print(rho)
            mean_n_t = observable.dot(rho).trace()
            #print(mean_n_t)
            Mean_n_t.append(mean_n_t)
            
        return Mean_n_t, T
        
    


