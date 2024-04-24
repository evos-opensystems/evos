#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:36:18 2022
@author: reka
"""
import numpy as np 
from ferm_bos_lattice.py import Operator as Op
from abc import ABC, abstractmethod


class FermionicState(): 
    def __init__(self): 
         k = 0
    def state_ket(self, N: int,  spin: int, alternate_spin: str, excitation_up: int = 0, excitation_down: int = 0) -> np.ndarray:
        
        if spin is None: dim = 2**N 
        else: dim = 4**N    
        dim = self.dim
        
        
        def vac_ket(self, dim):
    
            state_bra = np.zeros((1, self.dim))
            for i in range(0,dim+1):
                
                if i == 0:
                    state_bra[0,i] = 1
                    
            state_ket = np.zeros((dim, 1))
            for i in range(0,dim+1):
                if i == 0:
                    state_ket[i,0] = 1
                    
            state_bra = np.conjugate(state_ket)
            rho = np.outer(state_ket, state_bra)     
            # print(state_bra)
            return state_ket
        
        #########################################################
        
        if alternate_spin == 'alternate_spin': 
            
            # state with alternate up down fermions
            updown_ket = vac_ket(dim)   
            for i in np.arange(2,N+1,2):
                #updown_ket = np.dot(c_up_dag(i-1, N), updown_ket)
                updown_ket = np.dot(Op.fermion('adag', i-1,N,'up'), updown_ket)
                #updown_ket = np.dot(c_down_dag(i, N), updown_ket)
                updown_ket = np.dot(Op.fermion('adag', i,N,'down'), updown_ket)
                
        updown_bra = np.conjugate(updown_ket) 
           
        rho_updown = np.outer(updown_ket, updown_bra)
        
state = FermionicState()
state.state_ket(2, 1, 'alternate_spin')