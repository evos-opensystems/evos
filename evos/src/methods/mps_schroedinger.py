import numpy as np
import os
import sys
import time
from numpy import linalg as LA
from scipy.linalg import expm
import psutil
import pyten as ptn

class MPSSchroedinger():
    """_summary_
    """
    def __init__(self, n_sites: int, lat: ptn.mp.lat, H: ptn.mp.MPO, lindbl_op_list: list):
        """Set the number of sites and Hamiltonian to instance variable

        Parameters
        ----------
        n_sites : int
            number of lattice sites
        H : ptn.mp.MPO
            Hamiltonian (the hermitian, not the effective one)
        """
        
        self.n_sites = n_sites
        self.H = H
        

    def quantum_jump_single_trajectory_time_evolution(self, psi_t: ptn.mp.MPS, obsdict: dict, tdvp_config_list: list, krylov = True, krylov_config = ptn.krylov.Conf() ):
        """Compute the time-evolution via the quantum jumps method for a single trajectory. Two arrays r1 and r2 of random numbers are used 
        first to check if a jump needs to be applied if yes then which operator to use.

        Parameters UPDATE THEM!
        ----------
        psi_t : ptn.mp.MPS
            initial state to be evolved
        t_max : float
            maximal evolution time
        dt : float
            timestep
        obsdict: 
            instance of the class  'evos.src.observables.observables.Observables()'
        """
        
        #FIXME add option for time-dependent Hamiltonian!
        
        #compute observables with intial state
        obsdict.compute_all_observables_at_one_timestep(psi_t, 0)
        
        #perform global krylov time-evolution
        krylov_time_index = 0 #define it here so that it can be used when krylov = False
        if krylov:
            n_krylov_steps = int (krylov_config.tend / krylov_config.dt )
            evolver = ptn.mp.krylov.Evolver_A( self.H.copy(), psi_t, krylov_config.copy() )
            
            for krylov_time_index in n_krylov_steps:
                evolver.evolve_in_subspace() #NOTE: krylov states are saved
            times =  round( ( krylov_time_index + 1 ) * krylov_config.dt ,4 ) #FIXME: rounding comma can depend on timestep
            if ( float.is_integer(times) ): #check whether time needs to be rounded for state name
                times = int(times)
            state_name = 'krylov_T-'+str(times)+'.mps'
            time.sleep(0.5) #FIXME small sleep is needed to allow the state to being written. could need to be larger for large systems
            psi_t = ptn.mp.MPS(state_name)
            obsdict.compute_all_observables_at_one_timestep(psi_t, krylov_time_index + 1) #compute observables
        
        
        #tdvp evolution: loop over the different tdvp configurations in 'tdvp_config_dict'
        total_time_index = 0
        for tdvp_config in tdvp_config_list:
            
            n_tdvp_steps = int (tdvp_config.tend / tdvp_config.dt ) # number of timesteps for this specific tdvp configuration
            worker = ptn.mp.tdvp.PTDVP(psi_t.copy(),[self.H.copy()],tdvp_config.copy()) 
            for i in n_tdvp_steps: #FIXME: 'i' is unused 
                total_time_index += 1
                worker.do_step()
                psi_t = worker.get_psi(False)
                obsdict.compute_all_observables_at_one_timestep(psi_t, krylov_time_index + total_time_index + 1) #compute observables
        
                
            
            
            
            
            
    
                