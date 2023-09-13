import numpy as np
import os
import sys
import time
from numpy import linalg as LA
from scipy.linalg import expm
import psutil
import pyten as ptn

class MPSSchroedinger():
    """The method 'schroedinger_time_evolution' allows to concatenate a 
   global Krylov time-evolution with pyten with an arbitrary number of tdvp configurations in which one can change, for instance,
   the tdvp mode or the timestep. The krylov time evolution can be skipped by setting krylov=False.
   The main imput parameter of 'schroedinger_time_evolution' is 'tdvp_config_list', i.e. a list of ptn.tdvp.Conf() objects with the
   desired parameters.
    """
    def __init__(self, n_sites: int, lat: ptn.mp.lat, H: ptn.mp.MPO):
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
        

        # Parameters UPDATE THEM!!!!!!!!!!
        # ----------
        # psi_t : ptn.mp.MPS
        #     initial state to be evolved
        # t_max : float
        #     maximal evolution time
        # dt : float
        #     timestep
        # obsdict: 
        #     instance of the class  'evos.src.observables.observables.Observables()'
        # """
    def schroedinger_time_evolution(self, psi_t: ptn.mp.MPS, obsdict: dict, tdvp_config_list: list, krylov = True, krylov_config = ptn.krylov.Conf(), save_states = False ):
        """Compute unitary, real-time evolution for a time-independent Hamiltonian by concatenating a global Krylov time-evolution 
        with pyten with an arbitrary number of tdvp configurations in which one can change, for instance,
        the tdvp mode or the timestep.

        Parameters
        ----------
        psi_t : ptn.mp.MPS
            initial state to be evolved
        obsdict : dict
            instance of the class  'evos.src.observables.observables.Observables()'
        tdvp_config_list : list
            list of ptn.tdvp.Conf() objects
        krylov : bool, optional
            if =False, the global Krylov loop is skipped. By default True
        krylov_config : pyten.cpp_pyten.krylov.Conf, optional
            krylov configuration object, by default ptn.krylov.Conf()
        save_states : bool, optional
            if = True, all the krylov and tdvp states are saved. By default False    
        """        
        
        
        #FIXME add option for time-dependent Hamiltonian!
        
        #compute observables with intial state
        obsdict.compute_all_observables_at_one_timestep(psi_t, 0)
        
        #perform global krylov time-evolution
        krylov_time_index = 0 #define it here so that it can be used when krylov = False
        if krylov:
            print('krylov_config.tend= ', krylov_config.tend)
            print('krylov_config.dt =', krylov_config.dt)
            n_krylov_steps = int ( np.real(krylov_config.tend) / np.real(krylov_config.dt) )
            evolver = ptn.mp.krylov.Evolver_A( self.H.copy(), psi_t, krylov_config.copy() )
            
            for krylov_time_index in range(n_krylov_steps):
                evolver.evolve_in_subspace() #NOTE: krylov states are saved
                times =  round( ( krylov_time_index + 1 ) * np.real(krylov_config.dt) ,4 ) #FIXME: rounding comma can depend on timestep
                if ( float.is_integer(times) ): #check whether time needs to be rounded for state name
                    times = int(times)
                state_name = 'krylov_T-'+str(times)+'.mps'
                time.sleep(0.5) #FIXME small sleep is needed to allow the state to being written. could need to be larger for large systems
                psi_t = ptn.mp.MPS(state_name)
                obsdict.compute_all_observables_at_one_timestep(psi_t, krylov_time_index + 1) #compute observables
                time.sleep(0.1)
                if not save_states: 
                    os.remove(state_name) #remove state
        
        #tdvp evolution: loop over the different tdvp configurations in 'tdvp_config_dict'
        total_time_index = 0
        for tdvp_config in tdvp_config_list:
            print('switched to new tdvp config' )
            n_tdvp_steps = int ( np.real(tdvp_config.maxt) / np.real(tdvp_config.dt) ) # number of timesteps for this specific tdvp configuration
            worker = ptn.mp.tdvp.PTDVP(psi_t.copy(),[self.H.copy()],tdvp_config.copy()) 
            for i in range(n_tdvp_steps): #FIXME: 'i' is unused 
                total_time_index += 1
                worker.do_step()
                psi_t = worker.get_psi(False)
                obsdict.compute_all_observables_at_one_timestep(psi_t, krylov_time_index + total_time_index) #compute observables
                if save_states:
                    psi_t.save('state_t_' + str( round( (krylov_time_index + total_time_index) *  np.real(tdvp_config.dt), 4 ) ) ) 
                
            
            
            
            
            
    
                