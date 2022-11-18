#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

from abc import ABC, abstractmethod
import numpy as np





class SpinfullFermionsLattice:
    
    def __init__(self, N_sites: int):
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
        
        self.N_sites = N_sites


    def sso(self, operator_name: str, k: int, spin: str) -> np.ndarray:
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
        i = self.N_sites-k
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
        
   

    
    
