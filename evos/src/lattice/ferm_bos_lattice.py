#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

import numpy as np
    

class Operator:
    
    def __init__(self): 
        #operators        
        
        
        up = 'up'
        self.up = up
        down = 'down'
        self.down = down
        spin = 'spin'
        self.spin = spin
        #parity operator to the right Jordan Wigner Trafo
        P_r = np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
        self.P_r = P_r
        P = np.array([[1, 0], [0,-1]])
        self.P = P
        
    
    def fermion(self, operator_name: str, k:float, N:int, spin:str) -> np.ndarray:
        P = np.array([[ 1,0],[ 0,-1]]) #parity
        iden = np.array([[ 1,0],[ 0,1]])
        
        if spin == 'up':
            operator = np.kron(operator_name, P) #annihilate/create spin up
            
        if spin == 'down':   
            operator = np.kron( iden, np.dot(operator_name, iden)) #annihilate/create spin down
        
        if spin == None:
            operator = operator_name
        
        if spin == 'up' or spin == 'down':
            iden_l = np.zeros((4**(k-1), 4**(k-1)))
            for i in range(0, 4**(k-1)):
                for j in range(0, 4**(k-1)):
                    if i == j:
                        iden_l[i,j] = 1        
            i = N-k
            if i == 0:
                    P_r1 = 1
            if i == 1:
                P_r1 = self.P_r
            #parity operators to the right
            if i > 1: 
                P_r1 = self.P_r
                for l in range(0,i-1):
                    P_r1 = np.kron(P_r1, self.P_r*(l+1)/(l+1))
            a = np.kron(iden_l, np.kron(operator, P_r1))
            return a
        
        if spin == None: 
            iden_l = np.zeros((2**(k-1), 2**(k-1)))
            for i in range(0, 2**(k-1)):
                for j in range(0, 2**(k-1)):
                    if i == j:
                        iden_l[i,j] = 1        
            i = N-k
            if i == 0:
                    P_r1 = 1
            if i == 1:
                P_r1 = self.P
            #parity operators to the right
            if i > 1: 
                P_r1 = self.P
                for l in range(0,i-1):
                    P_r1 = np.kron(P_r1, self.P*(l+1)/(l+1))
            a = np.kron(iden_l, np.kron(operator, P_r1))
            return a
            
            

    def boson(self, operator_name: str, k:float, N:int) -> np.ndarray:
        iden_l = np.zeros((2**(k-1), 2**(k-1)))
        for i in range(0, 2**(k-1)):
            for j in range(0, 2**(k-1)):
                if i == j:
                    iden_l[i,j] = 1  
                    
        iden_r = np.zeros((2**(N-k), 2**(N-k)))
        for i in range(0, 2**(N-k)):
            for j in range(0, 2**(N-k)):
                if i == j:
                    iden_r[i,j] = 1 
                    
        a = np.kron(iden_l, np.kron(operator_name, iden_r))
        return a

# for example: adag = np.array([[ 0,0],[1,0]])
#                 a = np.array([[ 0,1],[0,0]])          
a = np.array([[ 0,1],[0,0]]) 
adag = np.array([[ 0,0],[1,0]])
      

op = Operator()

o = op.fermion(adag,1,1,spin = None)
b = op.boson(a,1,4)

print(b)



