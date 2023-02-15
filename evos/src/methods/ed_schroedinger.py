import numpy as np
import os
import sys
import time
from numpy import linalg as LA
from scipy.linalg import expm

class EdSchroedinger():
    """_summary_
    """
    def __init__(self, n_sites: int, H: np.ndarray):
        """Sets the number of site and the Hamiltonian to instance variables

        Parameters
        ----------
        n_sites : int
            number of lattice sites
        H : np.ndarray
            Hamiltonian
        """
        self.n_sites = n_sites
        self.H = H
        
        
    def schroedinger_time_evolution(self, psi_t: np.ndarray, t_max: float, dt: float, obsdict):
        """Compute the Schrödinger dynamics for the given Hamiltonian, initial state, observables and time interval

        Parameters
        ----------
        psi_t : np.ndarray
            initial state to be evolved
        t_max : float
            maximal evolution time
        dt : float
            timestep
        obsdict: 
            instance of the class  'evos.src.observables.observables.Observables()'
        """
        
        n_timesteps = int(t_max/dt) #NOTE: read from instance or compute elsewhere
        U = expm( -1j * dt * self.H ) #compute the time-evolution operator. needs to be done only once
        #compute observables with initiql state
        obsdict.compute_all_observables_at_one_timestep(psi_t, 0)        
        #loop over timesteps
        for i in range( n_timesteps ):
            #print('computing timestep ',i)
            psi_t = np.dot( U, psi_t.copy() )
            #Compute observables
            obsdict.compute_all_observables_at_one_timestep(psi_t, i+1) 