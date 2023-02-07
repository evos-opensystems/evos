#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:45:33 2023

@author: reka
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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


class SolveLindbladEquation():
    def __init__(self,dim_H, H, L_list, dt, T):
        self.dim_H = dim_H
        self.dt = dt
        self.T = T
        tsteps = int(self.T/self.dt)
        t = np.linspace(0,self.T, tsteps)
        self.tsteps = tsteps
        self.t = t
        
       
        self.H = H
        H1 = []
        for t11 in t:
           H1.append(H(t11))   
        self.H1 = H1
        
        self.L_list = L_list
        
    def solve(self, observable, rho_ket):
        rho_bra = np.conjugate(rho_ket) 
        rho_matrix = np.outer(rho_ket, rho_bra)     
    
        rho_vec = []
        for i in range(0, self.dim_H):
            for  j in range(0 , self.dim_H):
                rho_vec.append(rho_matrix[i,j])        
        rho_vec = np.array(rho_vec,dtype='complex')
        
        init_state = rho_vec
        
        Dyn_rho_dt = []
        #print(t)
        for t1 in range(len(self.t)):
            
            dyn = LindbladEquation(self.dim_H, self.H1[t1], self.L_list)
            Dyn_rho_dt.append(dyn)
            
        print(Dyn_rho_dt)
        
        #dyn = ed_mesoscopic_leads.MesoscopicLeadsLindblad(dim_H, H1[0], L_list_left)
        
        #sol = solve_ivp(dyn.drho_dt, (0,T) , rho_vec, t_eval = t)
        
        exp_n = []
        t11 = []
        
        t_before = 0
        for t1 in self.t:
            if t1 == 0:
                t_before =0
                exp = 1#observable.dot(rho_updown).trace()
                #rho_vec = rho_vec
            else:
                t_before_index = np.where(self.t==t1)[0] -1
                t_before = float(self.t[t_before_index])
            #print(int(np.where(t == t_before)[0]))
            
                dyn_drho_dt = Dyn_rho_dt[int(np.where(self.t == t_before)[0])]
                #print(dyn_drho_dt.drho_dt)
                #print(t1, t1+dt)
                
                sol = solve_ivp(dyn_drho_dt.drho_dt, (t_before,t_before+self.dt), rho_vec, t_eval = [t_before+self.dt])
            #print(np.shape(sol.y))
                
        
                #plot some expectation value at each time step
                #time dependant rho:
                rho_sol = np.zeros((self.dim_H,self.dim_H),dtype='complex')
                count=0
                for n in range(self.dim_H):
                    for  m in range(0,self.dim_H):
                        rho_sol[n,m] = sol.y[count,:]
                        count+=1
                    
                for n in range(self.dim_H):
                    for  m in range(0,self.dim_H):
                        rho_sol[n,m] = np.conjugate(rho_sol[m,n])
                        
                #print(sol.y[:,0])
                #compute expectation value
                rho_vec = sol.y[:,0]
            
                exp = observable.dot(rho_sol[:,:]).trace()
            exp_n.append(exp)
            t11.append(t_before)
            
        return exp_n, t11
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    