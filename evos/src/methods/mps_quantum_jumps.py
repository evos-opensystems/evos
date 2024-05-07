import numpy as np
import os
import sys
import time
from numpy import linalg as LA
import psutil
import pyten as ptn
##import evos.src.observables.observables as observables

class MPSQuantumJumps():
    """_summary_
    """
    def __init__(self, n_sites: int, lat: ptn.mp.lat, H: ptn.mp.MPO, lindbl_op_list: list, max_exp_sweep = 4, max_opt_sweep = 4):
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
        self.LdL = []
        H_eff = H.copy() #compute effective Hamiltonian H_eff = H - i/2 \sum_m L^\dagger _m * L_m 
        for i in range( len(lindbl_op_list) ):
            H_eff += - 0.5j * lindbl_op_list[i] * ptn.mp.dot( lat.get("I"), lindbl_op_list[i].copy() )  #NOTE: in pyten order of operators is reversed
            H_eff.truncate()
            self.LdL.append(lindbl_op_list[i] * ptn.mp.dot( lat.get("I"), lindbl_op_list[i].copy() ))

        H_eff_dag = ptn.mp.dot(lat.get("I"), H_eff.copy())
        H_s = 0.5 * ( H_eff.copy() + H_eff_dag.copy() ) #herm part
        H_a = 0.5 * ( H_eff.copy() - H_eff_dag.copy() ) #antiherm part
        H_as = -1j * H_a #make it herm
    
        self.H_eff = H_eff
        self.H_s = H_s
        self.H_as = H_as

        #param for applying jump operators
        self.max_exp_sweep = max_exp_sweep
        self.max_opt_sweep = max_opt_sweep
        
        
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


        # t1 = time.time()
        # for jump_op in self.lindbl_op_list:
        #     states_after_jump_operator_application = ptn.mp.apply_op_fit( psi.copy(), jump_op,  ptn.Truncation( threshold, maxStates, maxStates, weight ).scaled(norm), threshold, self.max_exp_sweep, self.max_opt_sweep)[0] #ptn.Truncation()
        #     states_after_jump_operator_application_list.append( states_after_jump_operator_application )
        # norms_after_jump_operator_application_vector_squared = np.zeros( len( states_after_jump_operator_application_list ) )
        # t2  = time.time()
        # print('time of all jumps application {:.3f}s'.format(t2-t1))
        # t1 = time.time()
        # for i in range( len( states_after_jump_operator_application_list ) ):
        #     norms_after_jump_operator_application_vector_squared[i] = states_after_jump_operator_application_list[i].norm() ** 2
        # t2 = time.time()
        # print('time of all jumps application norm {:.3f}s'.format(t2-t1))
        #Zhaoxuan: can be replaced by expectation instead of norm of MPO-MPS application
        t1 = time.time()
        norms_squared_exp = np.zeros( len(self.LdL) )
        for i in range( len( self.LdL ) ):
            norms_squared_exp[i] = ptn.mp.expectation(psi.copy(), self.LdL[i]).real
        t2 = time.time()
        print('time of all jumps expectation {:.3f}s'.format(t2-t1))
        # print("==norms square of application==")
        # print(norms_after_jump_operator_application_vector_squared)
        print("==norms from expectation==")
        print(norms_squared_exp)




        # tot_norm = sum(norms_after_jump_operator_application_vector_squared)
        tot_norm = sum(norms_squared_exp)
        #Normalize the probabilities
        # norms_after_jump_operator_application_vector_squared /= tot_norm
        norms_squared_exp /= tot_norm

        #make array with intervals proportional to probability of one jump occurring
        # intervals = np.zeros(len(states_after_jump_operator_application_list)+1)
        intervals = np.zeros(len(self.LdL)+1)
        # intervals[1] = norms_after_jump_operator_application_vector_squared[0]
        intervals[1] = norms_squared_exp[0]
        for i in range( 2, len(intervals ) ):
            # intervals[i] = intervals[i-1] + norms_after_jump_operator_application_vector_squared[i-1]
            intervals[i] = intervals[i-1] + norms_squared_exp[i-1]
        
        print('interval ', intervals)    
    
        #choose and apply jump operator 
        for i in range( 1,len( intervals ) ):
            if r2 >= intervals[i-1] and r2 <= intervals[i]:
                print(r2,"belongs to interval ",i, "that goes from ",intervals[i-1],"to",intervals[i])
                #test
                print("calculating jump operator application")
                t1 = time.time()
                # psi_test = ptn.mp.apply_op_fit( psi.copy(), self.lindbl_op_list[i-1],  ptn.Truncation( threshold, maxStates, maxStates, weight ).scaled(norm), threshold, self.max_exp_sweep, self.max_opt_sweep)[0]
                psi = ptn.mp.apply_op_fit( psi.copy(), self.lindbl_op_list[i-1],  ptn.Truncation( threshold, maxStates, maxStates, weight ).scaled(norm), threshold, self.max_exp_sweep, self.max_opt_sweep)[0]
                t2 = time.time()
                print('time of one jump application {:.3f}s'.format(t2-t1))
                #origin
                # psi = states_after_jump_operator_application_list[i-1]
                which_jump_op = i-1
                print('which jump = ', which_jump_op)
                norm_after_jump = psi.norm()
                print('norm_after_jump = ', norm_after_jump)

                # print('ovlp <app | exp> = {}'.format(ptn.mp.overlap(psi, psi_test)))
                # print('norm |app|^2 = {}'.format(psi.norm()**2))
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
        norm_evolution = []
        jump_prob = []
        
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
            
            t1 = time.time()
            psi_0 = worker.get_psi(False)
            norm_psi0 = psi_0.norm()
            print("norm before do_step() = {}".format(norm_psi0))
            worker_do_stepList = worker.do_step()
            psi_1 = worker.get_psi(False)
            
            norm_psi1 = psi_1.norm()
            print("norm after do_step() = {}".format(norm_psi1))
            delta_p = 1 - norm_psi1 ** 2
            print("jump probability = {}".format(delta_p))
            t2 = time.time()
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
            #debug
            norm_evolution.append(norm_psi1)
            np.savetxt('norm_evolution', norm_evolution)
            jump_prob.append(delta_p)
            np.savetxt('jump_prob', jump_prob)

            #normalize state
            psi_t.normalise()

            t3 = time.time()
            te = (t2 - t1) / (60 * 60)
            tj = (t3 - t2) / (60 * 60)
            print("time of evolution = {:.4f}hrs; time of jump = {:.4f}hrs".format(te, tj))
            
            #compute observables
            obsdict.compute_all_observables_at_one_timestep(psi_t, i+1) 
        # #at the end
        os.chdir('..') #exit the trajectory directory
        
