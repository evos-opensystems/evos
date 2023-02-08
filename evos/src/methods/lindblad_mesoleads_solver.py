#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice

class LindbladEquation: 
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
        
        
        
class MesoscopicLeads():
    def __init__(self, n_tot:float, n_lead_left:float, n_lead_right:float, T_L:float, T_R:float, mu_L:float, mu_R:float, T:float, dt:float, eps_vec_l:float, eps_vec_r:float, k_vec_L:float, k_vec_R:float):
        
        self.n_tot = n_tot 
        self.n_lead_left = n_lead_left
        self.n_lead_right = n_lead_right
        n_sites = n_tot - (n_lead_left + n_lead_right)
        self.n_sites = n_sites
        
        dim_tot = 4**n_tot
        self.dim_tot = dim_tot
        
        self.T_L = T_L
        self.T_R = T_R
        
        self.mu_L = mu_L
        self.mu_R = mu_R
        
        self.T = T
        self.dt = dt 
        
        self.eps_vec_l = eps_vec_l
        self.eps_vec_r = eps_vec_r
        
        self.k_vec_L = k_vec_L
        self.k_vec_R = k_vec_R
        
        
    def H_leads_left(self, eps, k_vec, mu_L):
        spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(self.n_tot)
        # LEAD SITES - kinetic energy of left leads
        kin_leads = np.zeros((self.dim_tot, self.dim_tot))
        if self.n_lead_left == 0: 
            print('Ekin_lead left terms on sites:', 0)
            for k in range(0, self.dim_tot):
                kin_leads = np.zeros((self.dim_tot, self.dim_tot))
        else: 
            for k in range(1, self.n_lead_left+1): 
                print('Ekin_lead left terms on sites:', k)
                kin_leads += (eps[k-1] - self.mu_L) *( np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up')) + np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down')))
         
   
        # HOPPING BETWEEN LEADS AND SYSTEM LEFT SIDE
        hop_sys_lead = np.zeros((self.dim_tot, self.dim_tot))
        if self.n_lead_left == 0: 
            print('left sys lead hopping on sites:', 0)
            for k in range(0, self.dim_tot): 
                hop_sys_lead = np.zeros((self.dim_tot, self.dim_tot))
        else: 
            for k in range(self.n_lead_left, self.n_lead_left+1): 
                print('left sys lead hopping on sites:', k, k+1)
                hop_sys_lead += k_vec[k-1]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
         
            
        H = kin_leads + hop_sys_lead    
        return H
        
        
        
        
        







