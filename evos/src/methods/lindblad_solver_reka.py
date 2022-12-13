#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class MesoscopicLeadsLindblad: 
    def __init__(self, dim_H:int, H:float, L_list:float):
        
        self.dim_H = dim_H
        self.H = H
        
        self.L_list = L_list
        #self.L_list_right = L_list_right
    
    def drho_dt(self, t, rho_vec):
            
        
            def commutator(x,y):
                return  np.dot(x,y) - np.dot(y,x)
            
            def anticom(x,y):
                return x.dot(y) + y.dot(x)
            
            
            #make matrix out of list
            rho = np.zeros((self.dim_H,self.dim_H),dtype="complex")
            count=0
            for n in range(0, self.dim_H):
                for m in range(0, self.dim_H):
                    rho[n,m] = rho_vec[count]
                    count+=1
                    
            for n in range(self.dim_H):
                for  m in range(0, self.dim_H):
                    rho[n,m] = np.conjugate(rho[m,n])
                    
            #print(rho)
            #Lindblad equation 
            #when there are individually different Lindblad operators on each site or something then use this one:
             
            
            drho = -1j* commutator(self.H, rho) 
            #print(drho)
            #for i in range(1,N+1): 
                #print(i)
                #drho += - 1/2*anticom(L(i, N).conj().T.dot(L(i,N)), rho) + L(i,N).dot(rho).dot(L(i,N).conj().T)
                
            for i in range(0, len(self.L_list)): 
                    drho += - 1/2*anticom(self.L_list[i].conj().T.dot(self.L_list[i]), rho) + self.L_list[i].dot(rho).dot(self.L_list[i].conj().T)
                    
            #print(self.H)
            #when its the same Lindblad operator on each site and there are many, use this one: 
            #drho = -1j* commutator(H,rho) - 1/2*anticom(L.conj().T.dot(L), rho)
                
            
            #turn back: make list out of matrix
            drho_l = []
            for n in range(0, self.dim_H):
                for  m in range(0, self.dim_H):
                    drho_l.append(drho[n,m])  
                    
            #print(sum(drho_l))     
            
            return drho_l







