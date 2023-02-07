#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class TimeDepHamiltonianLindblad: 
    def __init__(self, dim_H:int, H:float, dt:float, T:float):
        
        self.dt = dt
        self.T = T
        tsteps = int(self.T/self.dt)
        t = np.linspace(0,self.T, tsteps)
        self.tsteps = tsteps
        self.t = t
        
        
        self.dim_H = dim_H
        self.H = H
        H1 = []
        for t11 in t:
           H1.append(H(t11))   
        self.H1 = H1
        
        
    def solve_lindblad_equation(self, rho_ket, H, L_list, observable:float):
        
        def drho_dt(t, rho_vec, H, L_list):
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
                    
                    
            drho = -1j* commutator(np.array(H), rho) 
          
            for i in range(0, len(L_list)): 
                    drho += - 1/2*anticom(L_list[i].conj().T.dot(L_list[i]), rho) + L_list[i].dot(rho).dot(L_list[i].conj().T)
                    
           
            #turn back: make list out of matrix
            drho_l = []
            for n in range(0, self.dim_H):
                for  m in range(0, self.dim_H):
                    drho_l.append(drho[n,m])  
                    
            #print(sum(drho_l))     
            
            return drho_l  
        
        
        from scipy.integrate import solve_ivp

        
        rho_bra = np.conjugate(rho_ket) 
        rho_matrix = np.outer(rho_ket, rho_bra)     
    
        rho_vec = []
        for i in range(0, self.dim_H):
            for  j in range(0 , self.dim_H):
                rho_vec.append(rho_matrix[i,j])        
        rho_vec = np.array(rho_vec,dtype='complex')
        
        init_state = rho_vec
        
        Dyn_rho_dt = []
        for t1 in range(len(self.t)):
            H = self.H1[t1]
            t = t1
            dyn = drho_dt
            print(dyn)
           
            Dyn_rho_dt.append(dyn)
        #print('len Dyn_rho_dt =', len(Dyn_rho_dt))
       # print('H1 = ', self.H1)
        #print(t)    
        exp_n = []
        t11 = []
        
        t_before = 0
        for t1 in np.linspace(0,self.T, self.tsteps):
            t11.append(t1)
            
            if t1 == 0:
                t_before =0
                exp = observable.dot(rho_matrix).trace()
                init_state = rho_vec
                #print(t1)
            else:
                
                
                t_before_index = np.where(self.t==t1)[0] -1
                t_before = float(self.t[t_before_index])
                #print(int(np.where(t == t_before)[0]))
            
                dyn_drho_dt = Dyn_rho_dt[int(np.where(self.t == t_before)[0])]
                #print(dyn_drho_dt.drho_dt)
                #print(t1, t1+dt)
                #print('t1 =', t1)
                #print(dyn_drho_dt)
                '''
                sol = solve_ivp(dyn_drho_dt, (t_before,t_before+self.dt), rho_vec, t_eval = [t_before+self.dt])
                #print(np.shape(sol.y))
            
                #print(t1)
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
                        
                print(sol.y[:,0])
                #compute expectation value
                init_state = sol.y[:,0]
            
                exp = observable.dot(rho_sol[:,:]).trace()
                '''
                exp_n.append(exp)
                t11.append(t_before)
                
        return t11, exp_n
       
        
        