#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: reka
"""

from abc import ABC, abstractmethod
import numpy as np





class BosonicLattice:
    
    def __init__(self, N_sites:int):
        # operators   
        
        adag = np.array([[0, 0], [1, 0]])
        self.adag = adag
  
        a = np.array([[0, 1], [0, 0]])
        self.a = a
        
        iden = np.array([[1, 0], [0, 1]])
        self.iden = iden
        
        self.N_sites = N_sites

    def sso(self, operator_name: str, k:int, M:int) -> np.ndarray:
        """single site operator acting on Hilbert space of whole lattice, creating/annihilating 
        bosonic modes. 
    
        Args:
            operator_name (str):two options: 'adag' (creation) and 'a'
            (annihilation)
            M (int): number of modes
     
        Returns:
            np.ndarray: 
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
                    
        iden_r = np.zeros((2**(self.N_sites-k), 2**(self.N_sites-k)))
        for i in range(0, 2**(self.N_sites-k)):
            for j in range(0, 2**(self.N_sites-k)):
                if i == j:
                    iden_r[i, j] = 1
                    
        a = np.kron(iden_l, np.kron(operator, iden_r))
        return a