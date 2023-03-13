#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import evos.src.lattice.spinful_fermions_lattice as spinful_fermions_lattice

class LindbladEquation: 
    def __init__(self, n_tot:int, H:float, L_list:float):
        
        dim_H = 4**n_tot
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
    def __init__(self, n_tot:float, n_lead_left:float, n_lead_right:float, T_L:float, T_R:float, mu_L:float, mu_R:float, T:float, dt:float, eps_vec_l:float, eps_delta_vector_l:float, eps_vec_r:float, eps_delta_vector_r:float, k_vec_L:float, k_vec_R:float):
        
        self.n_tot = n_tot 
        self.n_lead_left = n_lead_left
        self.n_lead_right = n_lead_right
        n_sites = n_tot - (n_lead_left + n_lead_right)
        self.n_sites = n_sites
        
        dim_tot = 4**n_tot
        self.dim_tot = dim_tot
        
        #self.H_sys = H_sys
        
        self.T_L = T_L
        self.T_R = T_R
        
        self.mu_L = mu_L
        self.mu_R = mu_R
        
        self.T = T
        self.dt = dt 
        
        self.eps_vec_l = eps_vec_l
        self.eps_delta_vector_l = eps_delta_vector_l
        self.eps_vec_r = eps_vec_r
        self.eps_delta_vector_r = eps_delta_vector_r
        
        self.k_vec_L = k_vec_L
        self.k_vec_R = k_vec_R
        
        
    def H_leads_left(self):
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
                kin_leads += (self.eps_vec_l[k-1] - self.mu_L) *( np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up')) + np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down')))
         
   
        # HOPPING BETWEEN LEADS AND SYSTEM LEFT SIDE
        hop_sys_lead = np.zeros((self.dim_tot, self.dim_tot))
        if self.n_lead_left == 0: 
            print('left sys lead hopping on sites:', 0)
            for k in range(0, self.dim_tot): 
                hop_sys_lead = np.zeros((self.dim_tot, self.dim_tot))
        else: 
            for k in range(self.n_lead_left, self.n_lead_left+1): 
                print('left sys lead hopping on sites:', k, k+1)
                print(self.k_vec_L[k-1])
                hop_sys_lead += self.k_vec_L[k-1]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
        H =  hop_sys_lead + kin_leads 
        return H
            
         
            
    def H_leads_right(self):
        spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(self.n_tot)
        # LEAD SITES - kinetic energy of left leads
        kin_leads = np.zeros((self.dim_tot, self.dim_tot))
        if self.n_lead_right == 0: 
            print('Ekin_lead right terms on sites:', 0)
            for k in range(0, self.dim_tot): 
                kin_leads = np.zeros((self.dim_tot, self.dim_tot))
        else:       
            for k in range(self.n_tot - self.n_lead_right + 1, self.n_tot+1):    
                print('Ekin_lead right terms on sites:', k)
                kin_leads += (self.eps_vec_r[k - (self.n_tot - self.n_lead_right +1)] -self.mu_R) *( np.dot(spin_lat.sso('adag',k, 'up'), spin_lat.sso('a',k, 'up')) + np.dot(spin_lat.sso('adag',k, 'down'), spin_lat.sso('a',k, 'down')))
         
        # HOPPING BETWEEN LEADS AND SYSTEM RIGHT SIDE
        hop_sys_lead = np.zeros((self.dim_tot, self.dim_tot))
        if self.n_lead_right == 0: 
            print('right sys lead hopping on sites:', 0)
            for k in range(0, self.dim_tot):
                hop_sys_lead = np.zeros((self.dim_tot, self.dim_tot))
        else: 
            for k in range(self.n_tot - self.n_lead_right , self.n_tot): 
                print('right sys lead hopping on sites:', k, k+1)
                print(self.k_vec_R[k - (self.n_tot - self.n_lead_right)])
                hop_sys_lead += self.k_vec_R[k - (self.n_tot - self.n_lead_right) ]* (np.dot(spin_lat.sso('a',k, 'up'), spin_lat.sso('adag',k+1, 'up')) + np.dot(spin_lat.sso('a',k+1, 'up'), spin_lat.sso('adag',k, 'up')) + np.dot(spin_lat.sso('a',k, 'down'), spin_lat.sso('adag',k+1, 'down')) + np.dot(spin_lat.sso('a',k+1, 'down'), spin_lat.sso('adag',k, 'down')))
        H =  hop_sys_lead + kin_leads    
        return H
    
        #H = kin_leads + hop_sys_lead    
        #return H
    
    def lindbladlistmesoscopic(self):
        
        def fermi_dist(beta, e, mu):
            f = 1 / ( np.exp( beta * (e-mu) ) + 1)
            return f
        
        L_list = []
        spin_lat = spinful_fermions_lattice.SpinfulFermionsLattice(self.n_tot)
        for k in range(1, self.n_lead_left+1):
            print('k_left = ', k)
            print('epsdeltaL', self.eps_delta_vector_l[k-1])
            print('eps L', self.eps_vec_l[k-1])
            print('exponential left = ', np.exp( 1/self.T_L * (self.eps_vec_l[k-1] - self.mu_L) ))
            L_list.append( np.sqrt( self.eps_delta_vector_l[k-1]* np.exp( 1/self.T_L * (self.eps_vec_l[k-1] - self.mu_L) ) * fermi_dist(1/self.T_L, self.eps_vec_l[k-1], self.mu_L))* spin_lat.sso('a',k, 'up'))
            #print(spin_lat.sso('a', k, 'up'))
            L_list.append( np.sqrt( self.eps_delta_vector_l[k-1]* np.exp( 1/self.T_L * (self.eps_vec_l[k-1] - self.mu_L) ) * fermi_dist(1/self.T_L, self.eps_vec_l[k-1], self.mu_L)) * spin_lat.sso('a',k, 'down'))
            
            L_list.append( np.sqrt( self.eps_delta_vector_l[k-1]* fermi_dist(1/self.T_L, self.eps_vec_l[k-1], self.mu_L)) * spin_lat.sso('adag',k, 'up'))
            #print(spin_lat.sso('a', k, 'up'))
            L_list.append( np.sqrt( self.eps_delta_vector_l[k-1]* fermi_dist(1/self.T_L, self.eps_vec_l[k-1], self.mu_L)) * spin_lat.sso('adag',k, 'down'))
            
          
            
        for k in range(self.n_tot- self.n_lead_right +1, self.n_tot+1):
            print('k_right = ', k)
            print('epsdeltaR', self.eps_delta_vector_r[k-(self.n_tot- self.n_lead_right +1)])
            print('eps R', self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)])
            print('exponential right = ', np.exp( 1/self.T_R * (self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)] - self.mu_R) ))
            L_list.append( np.sqrt( self.eps_delta_vector_r[k-(self.n_tot- self.n_lead_right +1)]* np.exp( 1/self.T_R * (self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)] - self.mu_R) ) * fermi_dist(1/self.T_R, self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)], self.mu_R))* spin_lat.sso('a',k, 'up'))
            L_list.append( np.sqrt( self.eps_delta_vector_r[k-(self.n_tot- self.n_lead_right +1)]* np.exp( 1/self.T_R * (self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)] - self.mu_R) ) * fermi_dist(1/self.T_R, self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)], self.mu_R))* spin_lat.sso('a',k, 'down'))
            
            L_list.append( np.sqrt( self.eps_delta_vector_r[k-(self.n_tot- self.n_lead_right +1)]* fermi_dist(1/self.T_R, self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)], self.mu_R)) * spin_lat.sso('adag',k, 'up'))
            L_list.append( np.sqrt( self.eps_delta_vector_r[k-(self.n_tot- self.n_lead_right +1)]* fermi_dist(1/self.T_R, self.eps_vec_r[k-(self.n_tot- self.n_lead_right +1)], self.mu_R)) * spin_lat.sso('adag',k, 'down'))
            
        return L_list
    
    
class SolveLindblad():
    
    def __init__(self, n_tot:int):
        
        dim_tot = 4**n_tot 
        self.dim_tot = dim_tot
        
    def solve(self, init_state_ket:float, lindblad_equation:float, dt:float, T:float):
        
        tsteps = int(T/dt)
        t = np.linspace(0,T, tsteps)
        
        # make list out of matrix
        rho_bra = np.conjugate(init_state_ket) 
        rho_matrix = np.outer(init_state_ket, rho_bra)     
    
        rho_vec = []
        for i in range(0, self.dim_tot):
            for  j in range(0 , self.dim_tot):
                rho_vec.append(rho_matrix[i,j])        
        rho_vec = np.array(rho_vec,dtype='complex')
        
        #initital state
        init_state = rho_vec
        
        sol = solve_ivp(lindblad_equation, (0,T), rho_vec, t_eval=t)        
        #print(sol.y)
        
        
        #plot some expectation value at each time step
        #time dependant rho:
        rho_sol = np.zeros((self.dim_tot, self.dim_tot, tsteps),dtype='complex')
        count=0
        for n in range(self.dim_tot):
            for  m in range(0, self.dim_tot):
                rho_sol[n,m,:] = sol.y[count,:]
                count+=1
            
        for n in range(self.dim_tot):
            for  m in range(0, self.dim_tot):
                rho_sol[n,m,:] = np.conjugate(rho_sol[m,n])
                
        return rho_sol
                
            
        

    
            
            
        
        
        







