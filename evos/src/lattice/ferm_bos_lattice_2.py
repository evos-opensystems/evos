#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

from abc import ABC, abstractmethod
import numpy as np





class Sso:
    
    def __init__(self):
        # operators   
        
        adag = np.array([[0, 0], [1, 0]])
        self.adag = adag
  
        a = np.array([[0, 1], [0, 0]])
        self.a = a
        
        up = 'up'
        self.up = up
        down = 'down'
        self.down = down
        spin = 'spin'
        self.spin = spin
        # parity operator to the right Jordan Wigner Trafo
        P_r = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        self.P_r = P_r
        P = np.array([[1, 0], [0, -1]])
        self.P = P
        
        iden = np.array([[1, 0], [0, 1]])
        self.iden = iden
        


    def fermsp(self, operator_name: str, spin: str, k: int, N: int) -> np.ndarray:
        """ returns a fermionic (including spin 1/2 degrees
        of freedom: up and down) creation or annihilation operator acting on
        site k, identities on the rest of a lattice of N sites.

        Args:
            operator_name (str): two options: 'adag' (creation) and 'a'
            (annihilation)
            k (int): site on which operator is supposed to act on
            N (int): total number of sites in lattice

        Returns:
            np.ndarray: single site operator on site k, acting on whole Hilbert space
        """
        

        if operator_name == 'a':
            operator1 = self.a 
            
        if operator_name == 'adag':
            operator1 = self.adag
            
        if spin == 'up':
            operator = np.kron(operator1, self.P)
            # annihilate/create spin up
            
        if spin == 'down':
            operator = np.kron(self.iden, np.dot(operator1, self.iden))
            
                
        iden_l = np.zeros((4**(k-1), 4**(k-1)))
        for i in range(0, 4**(k-1)):
            for j in range(0, 4**(k-1)):
                if i == j:
                    iden_l[i, j] = 1
        i = N-k
        if i == 0:
                P_r1 = 1
        if i == 1:
            P_r1 = self.P_r
        # parity operators to the right
        if i > 1:
            P_r1 = self.P_r
            for l in range(0, i-1):
                P_r1 = np.kron(P_r1, self.P_r*(l+1)/(l+1))

        
        a1 = np.kron(iden_l, np.kron(operator, P_r1))

        return a1
        
    def ferm(self, operator_name: str, k: float, N: int) -> np.ndarray:
        
        if operator_name == 'a':
            operator1 = self.a
            
        if operator_name == 'adag':
            operator1 = self.adag
        
        operator = operator1
        
        iden_l = np.zeros((2**(k-1), 2**(k-1)))
        for i in range(0, 2**(k-1)):
            for j in range(0, 2**(k-1)):
                if i == j:
                    iden_l[i, j] = 1
        i = N-k
        if i == 0:
                P_r1 = 1
        if i == 1:
            P_r1 = self.P
        # parity operators to the right
        if i > 1: 
            P_r1 = self.P
            for l in range(0, i-1):
                P_r1 = np.kron(P_r1, self.P*(l+1)/(l+1))
        a = np.kron(iden_l, np.kron(operator, P_r1))
        return a

    
    
    
    def bos(self, operator_name: str, k:int, M:int, N:int) -> np.ndarray:
        """at the moment a single site operator, creating/annihilating 
        bosonic modes. 
    
        Args:
            operator_name (str):two options: 'adag' (creation) and 'a'
            (annihilation)
            M (int): number of modes
     
        Returns:
            np.ndarray: array representing a single site bosonic operator
            of mode M
        """
        
        def delta(k, l):
            return 1 if k == l else 0   
        
        if operator_name == 'a':
            operator = np.zeros((M+1, M+1))
            for n in range(0, M):
                for nn in range(0, M):
                    operator[n, nn+1] = np.sqrt(n+1)*delta(n, nn)
        
        if operator_name == 'adag':
            operator = np.zeros((M+1, M+1))
            for n in range(0, M):
                for nn in range(0, M):
                    operator[n+1, nn] = np.sqrt(nn+1)*delta(n, nn)
                    
                    
        iden_l = np.zeros((2**(k-1), 2**(k-1)))
        for i in range(0, 2**(k-1)):
            for j in range(0, 2**(k-1)):
                if i == j:
                    iden_l[i, j] = 1
                    
        iden_r = np.zeros((2**(N-k), 2**(N-k)))
        for i in range(0, 2**(N-k)):
            for j in range(0, 2**(N-k)):
                if i == j:
                    iden_r[i, j] = 1
                    
        a = np.kron(iden_l, np.kron(operator, iden_r))
        return a

o = Sso()
k = o.bos('adag', 1, 3, 2)

print(k)
