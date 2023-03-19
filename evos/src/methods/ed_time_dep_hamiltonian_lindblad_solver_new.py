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
        """computes the right side of the lindblad master equation

        Args:
            dim_H (int): dimension of total Hilbert space
            H (float): Hamiltonian (can be time dependant)
            L_list (float): list of Lindblad operators
        """
        
        self.dim_H = dim_H
        self.H = H
        self.L_list = L_list
        
    
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
                    
            # print('self.H ',self.H)
            # print('rho',rho)
            drho = -1j* commutator(self.H, rho) 
                
            for i in range(0, len(self.L_list)): 
                    drho += - 1/2*anticom(self.L_list[i].conj().T.dot(self.L_list[i]), rho) + self.L_list[i].dot(rho).dot(self.L_list[i].conj().T)

            #turn back: make list out of matrix
            drho_l = []
            for n in range(0, self.dim_H):
                for  m in range(0, self.dim_H):
                    drho_l.append(drho[n,m])  

            return drho_l


class SolveLindbladEquation():
    def __init__(self,dim_H:int, H:float, L_list:float, dt:float, T:float):
        """_summary_ method solve: takes an observable and an initial state, solves Lindblad equation containing time 
        dependant Hamiltonian and returns time dependant expectation value of observable as well as a list of the time steps

        Args:
            dim_H (int): dimension of Hilbert space
            H (float): Hamiltonian
            L_list (float): list of Lindblad operators
            dt (float): time step size
            T (float): final time
        """
        
        self.dim_H = dim_H
        self.dt = dt
        self.T = T
        tsteps = int(self.T/self.dt)
        t = np.arange(0, self.T, self.dt) #np.linspace(0,self.T, tsteps)
        self.tsteps = tsteps
        self.t = t
        
       
        self.H = H
        H1 = []
        for t11 in t:
           H1.append(H(t11))   
        self.H1 = H1
        
        self.L_list = L_list
        
    def solve(self, observable, rho_ket):
        
        # make list out of matrix
        rho_bra = np.conjugate(rho_ket) 
        rho_matrix = np.outer(rho_ket, rho_bra)     
    
        rho_vec = []
        for i in range(0, self.dim_H):
            for  j in range(0 , self.dim_H):
                rho_vec.append(rho_matrix[i,j])        
        rho_vec = np.array(rho_vec,dtype='complex')
        
        #initital state
        init_state = rho_vec
        
        # list of right sie of lindblad equation for each time step (time dependant Hamiltonian)
        Dyn_rho_dt = []
        for t1 in range(len(self.t)):
            
            dyn = LindbladEquation(self.dim_H, self.H1[t1], self.L_list)
            Dyn_rho_dt.append(dyn)
        
        exp_n = []
        t11 = []
        
        t_before = 0
        for t1 in self.t:
            if t1 == 0:
                t_before =0
                exp = 1 
                # print(t1)

            else:
                # print('timestep = ', t1)
                t_before_index = np.where(self.t==t1)[0] -1
                t_before = float(self.t[t_before_index])
            
                dyn_drho_dt = Dyn_rho_dt[int(np.where(self.t == t_before)[0])]
                # print(t1)
                

                sol = solve_ivp(dyn_drho_dt.drho_dt, (t_before,t_before+self.dt), rho_vec, t_eval = [t_before+self.dt])
            
                # solution into matrix
                rho_sol = np.zeros((self.dim_H,self.dim_H),dtype='complex')
                count=0
                for n in range(self.dim_H):
                    for  m in range(0,self.dim_H):
                        rho_sol[n,m] = sol.y[count,:]
                        count+=1
                    
                for n in range(self.dim_H):
                    for  m in range(0,self.dim_H):
                        rho_sol[n,m] = np.conjugate(rho_sol[m,n])
                        
                #assign new initial state for next time step
                rho_vec = sol.y[:,0]
                
                # expectation value of observable
                exp = observable.dot(rho_sol[:,:]).trace()
                
                # assign new initial state for next time step
                rho_vec = sol.y[:,0]
                
            exp_n.append(exp)
            t11.append(t_before)
            
        return exp_n, t11
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    