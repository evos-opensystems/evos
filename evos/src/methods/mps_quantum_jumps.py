import numpy as np
import os
import sys
import time
from numpy import linalg as LA
from scipy.linalg import expm
import psutil
import pyten as ptn
##import evos.src.observables.observables as observables

class MPSQuantumJumps():
    """_summary_
    """
    def __init__(self, n_sites: int, lat: ptn.mp.lat, H: ptn.mp.MPO, lindbl_op_list: list):
        """Computes the effective Hamiltonian and splits in into a Hermitian and and AntiHermitian part. Adds lindblad operators,
        effective hamiltonian and n_sites to instance variables.
        Similar init to that of the Lindblad class.

        Parameters
        ----------
        n_sites : int
            number of lattice sites
        H : ptn.mp.MPO
            Hamiltonian (the hermitian, not the effective one)
        lindbl_op_list : list
            list with the lindblad operators
        """
        
        self.n_sites = n_sites
        self.lindbl_op_list = lindbl_op_list
        self.lat = lat
        H_eff = H.copy() #compute effective Hamiltonian H_eff = H - i/2 \sum_m L^\dagger _m * L_m 
        for i in range( len(lindbl_op_list) ):
            H_eff += - 0.5j * lindbl_op_list[i] * ptn.mp.dot( lat.get("I"), lindbl_op_list[i].copy() )  #NOTE: in pyten order of operators is reversed
            H_eff.truncate()
        
        H_eff_dag = ptn.mp.dot(lat.get("I"), H_eff.copy())
        H_s = 0.5 * ( H_eff.copy() + H_eff_dag.copy() ) #herm part
        H_a = 0.5 * ( H_eff.copy() - H_eff_dag.copy() ) #antiherm part
        H_as = -1j * H_a #make it herm
    
        self.H_eff = H_eff
        self.H_s = H_s
        self.H_as = H_as
        
        
    def select_jump_operator(self, psi: ptn.mp.MPS, r2: float) -> tuple:
        """Selects which lindblad operator to apply for a jump out of the 'lindbl_op_list', given the state 'psi' and the pseudo-random number 'r2'

        Parameters
        ----------
        psi : ptn.mp.MPS
            state on which a jump operator need to be applied
        r2 : float
            pseudo-random number drawn from uniform distribution between 0 and 1.

        Returns
        -------
        tuple[ptn.mp.MPS, int]
            returns the state after the jump application and the index (integer) indicating which operator out of the input list hast been applied.
            The second output is only for debugging.
        """
        #get threshold, weight and maxStates from tdvp_config
        threshold = self.conf_tdvp.trunc.threshold
        weight = self.conf_tdvp.trunc.weight
        maxStates = self.conf_tdvp.trunc.maxStates
        
        states_after_jump_operator_application_list = []
        #rescale threshold, weight and maxStates by norm
        norm = psi.norm() 
        #threshold *= norm 
        #weight *= norm ** 2
        #maxStates = int(maxStates * norm ** 2 )
        for jump_op in self.lindbl_op_list:
            states_after_jump_operator_application = ptn.mp.apply_op_fit( psi.copy(), jump_op,  ptn.Truncation( threshold, maxStates, maxStates, weight ).scaled(norm), threshold, 12, 4)[0] #ptn.Truncation()
            states_after_jump_operator_application_list.append( states_after_jump_operator_application )

        norms_after_jump_operator_application_vector_squared = np.zeros( len( states_after_jump_operator_application_list ) )
        for i in range( len( states_after_jump_operator_application_list ) ):
            norms_after_jump_operator_application_vector_squared[i] = states_after_jump_operator_application_list[i].norm() ** 2

        tot_norm = sum(norms_after_jump_operator_application_vector_squared)
        #Normalize the probabilities
        norms_after_jump_operator_application_vector_squared /= tot_norm

        #make array with intervals proportional to probability of one jump occurring
        intervals = np.zeros(len(states_after_jump_operator_application_list)+1)
        intervals[1] = norms_after_jump_operator_application_vector_squared[0]
        for i in range( 2, len(intervals ) ):
            intervals[i] = intervals[i-1] + norms_after_jump_operator_application_vector_squared[i-1]
        
        print('interval ', intervals)    
    
        #choose and apply jump operator 
        for i in range( 1,len( intervals ) ):
            if r2 >= intervals[i-1] and r2 <= intervals[i]:
                print(r2,"belongs to interval ",i, "that goes from ",intervals[i-1],"to",intervals[i])
                psi = states_after_jump_operator_application_list[i-1]
                which_jump_op = i-1
                norm_after_jump = psi.norm()
                break
        print('After jump, norm(psi) = {}'.format( psi.norm() ) )    
        print('finished "select_jump_operator" method')    
        psi.normalise()
        print('The state after the jump has then be normalised')    
        return psi, which_jump_op, norm_after_jump     

    
    def quantum_jump_single_trajectory_time_evolution(self, psi_t: ptn.mp.MPS, conf_tdvp, t_max: float, dt: float, trajectory: int, obsdict: dict):
        
        """Compute the time-evolution via the quantum jumps method for a single trajectory. Two arrays r1 and r2 of random numbers are used 
        first to check if a jump needs to be applied if yes then which operator to use.

        Parameters
        ----------
        psi_t : ptn.mp.MPS
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
        
        self.conf_tdvp = conf_tdvp
        #non-hermitian tdvp #FIXME: specify this before
        self.conf_tdvp.exp_conf.mode = 'N'  #FIXME: specify this before
        self.conf_tdvp.exp_conf.submode = 'a' #FIXME: specify this before
        self.conf_tdvp.exp_conf.minIter = 15 #FIXME: specify this before
        worker = ptn.mp.tdvp.PTDVP( psi_t.copy(),[self.H_eff.copy()], self.conf_tdvp.copy() )
        
        os.mkdir( str( trajectory ) ) #create directory in which to run trajectory
        os.chdir( str( trajectory ) ) #change to it
        n_timesteps = int(t_max/dt) #NOTE: read from instance or compute elsewhere
        jump_counter = 0 #debugging
        jump_time_list = [] #debugging
        which_jump_op_list = [] #debugging
        r2_atjump_list = [] #debugging
        norm = []
        
        np.random.seed( trajectory + 1 ) #set seed for r1 this trajectory
        r1_array = np.random.uniform( 0, 1, n_timesteps ) #generate random numbers array r1
        #print('r1_array: ',r1_array)
        np.random.seed( int( ( trajectory + 1 ) / dt ) ) #set seed for r2 this trajectory
        r2_array = np.random.uniform( 0, 1, n_timesteps )  #generate random numbers array r2 to be used by method 'select_jump_operator()'

        #Compute observables with initial state
        obsdict.compute_all_observables_at_one_timestep(psi_t, 0)        
        #loop over timesteps
        memory_usage = []
        switched_to_1tdvp = False
        for i in range( n_timesteps ):
            r1 = r1_array[i] 
            #reinitialize worker with normalized state
            worker = ptn.mp.tdvp.PTDVP( psi_t.copy(),[self.H_eff.copy()], self.conf_tdvp.copy() ) 
            
            process = psutil.Process(os.getpid())
            memory_usage.append( process.memory_info().rss ) # in bytes
            np.savetxt('memory_usage', memory_usage)
            
            worker_do_stepList = worker.do_step()
            psi_1 = worker.get_psi(False)
            
            norm_psi1 = psi_1.norm()
             
            delta_p = 1 - norm_psi1 ** 2
            
            if r1 > delta_p: #evolve with non-hermitian hamiltonian
                psi_t = psi_1.copy()
            
            elif r1 <= delta_p: #select a lindblad operator and perform a jump
                print('jump occured at time', i*conf_tdvp.dt)
                psi_t, which_jump_op, norm_after_jump  = self.select_jump_operator( psi_t, r2_array[i] )   
                which_jump_op_list.append( which_jump_op ) #debugging
                jump_time_list.append(i * dt )
                jump_counter +=1 #debugging
                norm.append(norm_after_jump) # debugging
                np.savetxt('norm_after_jump', norm) #if trajectories run sequentially, this is being overwritten 
                np.savetxt('which_jump_op_list_last_trajectory',which_jump_op_list) #if trajectories run sequentially, this is being overwritten    
                np.savetxt('jump_time_list',jump_time_list) #if trajectories run sequentially, this is being overwritten    

            
            #normalize state
            psi_t.normalise()
            
            #compute observables
            obsdict.compute_all_observables_at_one_timestep(psi_t, i+1) 
        # #at the end
        os.chdir('..') #exit the trajectory directory
        
