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

        

    
    def fermion(self, operator_name: str, k: float, N: int, spin: str) -> np.ndarray:
        """returns a fermionic creation or annihilation operator acting on
        site k, identities on the rest of a lattice of N sites.

        Args:
            operator_name (str): two options: 'adag' (creation) and 'a'
            (annihilation)
            k (float): site on which operator is acting on
            N (int): total number of sites in lattice
            spin (str): can be "up", "down", or "None", depending if system is
            with spin, or spinless
        Returns:
            np.ndarray: operator on whole lattice
        """
        P = np.array([[1, 0], [0, -1]])  # parity
        iden = np.array([[1, 0], [0, 1]])
        
        if operator_name == 'a':
            operator_name1 = self.a
            
        if operator_name == 'adag':
            operator_name1 = self.adag
        
        if spin == 'up':
            operator = np.kron(operator_name1, P)
            # annihilate/create spin up
            
        if spin == 'down':
            operator = np.kron(iden, np.dot(operator_name1, iden))
            # annihilate/create spin down
        
        if spin is None:
            operator = operator_name1
        
        if spin == 'up' or spin == 'down':
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
        
        if spin is None: 
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
            
            

    def boson(self, operator_name: str, M:int) -> np.ndarray:
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
            a = np.zeros((M+1, M+1))
            for n in range(0, M):
                for nn in range(0, M):
                    a[n, nn+1] = np.sqrt(n+1)*delta(n, nn)
        
        if operator_name == 'adag':
            a = np.zeros((M+1, M+1))
            for n in range(0, M):
                for nn in range(0, M):
                    a[n+1, nn] = np.sqrt(nn+1)*delta(n, nn)
        return a


# examples :
o = Sso()
# fermionic creation operator on site k = 1, N = 2 total sites, with spin:
a = o.fermion('adag', 1, 1, spin='up')

# fermionic annihilation operator on site k = 2, total sites N = 2,
# in spinless system: 
b = o.fermion('a', 2, 2, spin=None)

# bosonic creation operator with 4 total modes:
c = o.boson('adag', 4)

# print('a= ', a)
# print('b=', b)
# print('c=', c)

unelectron = Sso()

print(a)


