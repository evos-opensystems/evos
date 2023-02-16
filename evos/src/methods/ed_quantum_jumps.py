import numpy as np
import os
import sys
import time
from numpy import linalg as LA
from scipy.linalg import expm
##import evos.src.observables.observables as observables

class EdQuantumJumps():
    """_summary_
    """
    def __init__(self, n_sites: int, H: np.ndarray, lindbl_op_list: list):
        """Computes the hermitian conjugate of the lindblad operators. Computes the effective Hamiltonian. Adds lindblad operators,
        effective hamiltonian and n_sites to instance variables.
        Very similar init to that of the Lindblad class.

        Parameters
        ----------
        n_sites : int
            number of lattice sites
        H : np.ndarray
            Hamiltonian (the hermitian, not the effective one)
        lindbl_op_list : list
            list with the lindblad operators
        """
        
        self.n_sites = n_sites
        
        lindbl_op_list_conj = []
        for op in lindbl_op_list:
            lindbl_op_list_conj.append( op.conj().T )
        self.lindbl_op_list = lindbl_op_list
        self.lindbl_op_list_conj = lindbl_op_list_conj
        
        H_eff = H.copy() #compute effective Hamiltonian H_eff = H - i/2 \sum_m L^\dagger _m * L_m 
        for i in range( len(lindbl_op_list) ):
            H_eff += - 0.5j * np.dot( lindbl_op_list_conj[i], lindbl_op_list[i] )
        self.H_eff = H_eff
        
    
<<<<<<< HEAD
    def select_jump_operator(self, psi: np.ndarray, r2: float) -> tuple([np.ndarray, int]) :
=======
    def select_jump_operator(self, psi: np.ndarray, r2: float) :
>>>>>>> reka
        """Selects which lindblad operator to apply for a jump out of the 'lindbl_op_list', given the state 'psi' and the pseudo-random number 'r2'

        Parameters
        ----------
        psi : np.ndarray
            state on which a jump operator need to be applied
        r2 : float
            pseudo-random number drawn from uniform distribution between 0 and 1.

        Returns
        -------
        tuple[np.ndarray, int]
            returns the state after the jump application and the index (integer) indicating which operator out of the input list hast been applied.
            The second output is only for debugging.
        """
        #cast all lindblad operators from numpy matrix to numpy array to be able to use np.dot!
        
        states_after_jump_operator_application_list = []
        for jump_op in self.lindbl_op_list:
            states_after_jump_operator_application_list.append( np.dot( jump_op, psi.copy( ) ) )

        norms_after_jump_operator_application_vector = np.zeros( len( states_after_jump_operator_application_list ) )
        for i in range( len( states_after_jump_operator_application_list ) ):
            norms_after_jump_operator_application_vector[i] = LA.norm( states_after_jump_operator_application_list[i] )

        tot_norm = sum(norms_after_jump_operator_application_vector)

        #Normalize the probabilities
        norms_after_jump_operator_application_vector /= tot_norm

        #make array with intervals proportional to probability of one jump occurring
        intervals = np.zeros(len(states_after_jump_operator_application_list)+1)
        intervals[1] = norms_after_jump_operator_application_vector[0]
        for i in range( 2, len(intervals ) ):
            intervals[i] = intervals[i-1] + norms_after_jump_operator_application_vector[i-1]
    
        #choose and apply jump operator 
        for i in range( 1,len( intervals ) ):
            if r2 >= intervals[i-1] and r2 <= intervals[i]:
                #print(r2,"belongs to interval ",i, "that goes from ",intervals[i-1],"to",intervals[i])
                psi = states_after_jump_operator_application_list[i-1]
                which_jump_op = i-1
                break
        return psi, which_jump_op      

    

    def quantum_jump_single_trajectory_time_evolution(self, psi_t: np.ndarray, t_max: float, dt: float, trajectory: int, obsdict):
        """Compute the time-evolution via the quantum jumps method for a single trajectory. Two arrays r1 and r2 of random numbers are used 
        first to check if a jump needs to be applied if yes then which operator to use.

        Parameters
        ----------
        psi_t : np.ndarray
            initial state to be evolved
        t_max : float
            maximal evolution time
        dt : float
            timestep
        trajectory : int
            integer labelling the trajectory
        obsdict: 
            instance of the class  'evos.src.observables.observables.Observables()'
        """
        os.mkdir( str( trajectory ) ) #create directory in which to run trajectory
        os.chdir( str( trajectory ) ) #change to it
        n_timesteps = int(t_max/dt) #NOTE: read from instance or compute elsewhere
        jump_counter = 0 #debugging
        jump_time_list = [] #debugging
        which_jump_op_list = [] #debugging
        r2_atjump_list = [] #debugging
        
        U = expm( -1j * dt * self.H_eff ) #compute the time-evolution operator. needs to be done only once
        np.random.seed( trajectory + 1 ) #set seed for r1 this trajectory
        r1_array = np.random.uniform( 0, 1, n_timesteps ) #generate random numbers array r1
        # print('r1_array: ',r1_array)
        np.random.seed( int( ( trajectory + 1 ) / dt ) ) #set seed for r2 this trajectory
        r2_array = np.random.uniform( 0, 1, n_timesteps )  #generate random numbers array r2 to be used by method 'select_jump_operator()'

        #Compute observables with initiql state
        obsdict.compute_all_observables_at_one_timestep(psi_t, 0)        
        #loop over timesteps
        for i in range( n_timesteps ):
            #print('computing timestep ',i)
            psi_1 = np.dot( U, psi_t.copy() )
            norm_psi1 = LA.norm( psi_1 )
            #print('norm_psi1 at timestep {} :'.format(norm_psi1, i))
            r1 = r1_array[i] 
            delta_p = 1 - norm_psi1 ** 2
            
            if r1 > delta_p: #evolve with non-hermitian hamiltonian
                psi_t = psi_1.copy()
            
            elif r1 <= delta_p: #select a lindblad operator and perform a jump
                #print('jump occured at timestep {}'.format(i)) #debugging
                jump_time_list.append(i) #debugging
                psi_t, which_jump_op  = self.select_jump_operator( psi_t, r2_array[i] )   
                which_jump_op_list.append( which_jump_op ) #debugging
                r2_atjump_list.append( r2_array[i] ) #debugging
                jump_counter +=1 #debugging
                #print('state after jump: ',psi_t)
                
            psi_t /= LA.norm( psi_t )

            #Compute observables
            #t_obs_start = time.process_time()
            obsdict.compute_all_observables_at_one_timestep(psi_t, i+1) 
            #print('process time for observables at timest {}: {}'.format(i, time.process_time() - t_obs_start) )
        os.chdir('..') #exit the trajectory directory
        #print('jump_counter: ',jump_counter)    
            
        
        
