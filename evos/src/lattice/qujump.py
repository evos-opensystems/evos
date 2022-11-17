#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

import numpy as np

class Solve():
    
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
            #print('r = ', r)
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
            #print(mean_n_t)
            Mean_n_t.append(mean_n_t)
            
        return Mean_n_t, T
        
    




