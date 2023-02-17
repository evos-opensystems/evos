import numpy as np

class ObservablesDict():
    
    def __init__(self):
        """Initialize two dictionaries, 'observables_array_dict' in which the computed observables are stored and 
        'observables_comp_functions_dict' in which the function that is used to compute the observable at each timestep is stored.
        They are both saved in the instance.
        """
        self.observables_array_dict = {}
        self.observables_comp_functions_dict = {}

    def initialize_observable( self, obs_name:str, obs_array_shape:tuple, n_timesteps:int, dtype='float' ):
        """Takes an observable name, the shape of the array in which to save it and the total number of steps in the time-evolution
        and adds it to the dictionray 'observables_array_dict'.

        Parameters
        ----------
        obs_name : str
            name of the observable
        obs_array_shape : tuple
            shape of the array in which to save the observable
        n_timesteps : int
            total number of timesteps for the time-evolution of interest
        dtype : str, optional
            data type of the numpy array in which the observable is stored, by default 'float'
        """
        obs_array_shape = sum( (obs_array_shape, (n_timesteps+1,) ), ()) 
        obs_array = np.zeros( obs_array_shape, dtype=dtype)

        self.observables_array_dict.update( {obs_name:obs_array} )

    def save_observables(self ):
        """Saves all the observables contained in the dictionary 'observables_array_dict'.
        NOTE: in numpy arrays of rank > 2 need to be pickled.
        """
        for key, value in self.observables_array_dict.items():
            #print(key, value)
            if len(value.shape) < 3:
                with open(key, 'wb') as f:
                    np.savetxt(f, value)
            if len(value.shape) >= 3:
                with open(key, 'wb') as f:
                    np.save(f, value)    

    def add_observable_computing_function(self,obs_name: str, observable_computing_function):
        """Assigns a function with which to compute the observable 'obs_name' at each timestep to the dictionary 'observables_comp_functions_dict'

        Parameters
        ----------
        obs_name : str
            name of the observable
        observable_computing_function : callable
            function with which to compute the observable 'obs_name' at each timestep
        """
        self.observables_comp_functions_dict.update( {obs_name:observable_computing_function} )


    def compute_all_observables_at_one_timestep(self, state, timestep: int, save=True):
        """_summary_

        Parameters
        ----------
        state : either ptn.mp.MPS or np.ndarry
            quantum state at time 'timestep'
        timestep : int
            timestep
        save : bool, optional
            save the observables at each timestep, by default True

        Raises
        ------
        ValueError
            class implemented only for rank 1,2,3 arrays
        """
        for observables_array_dict_key, observables_array_dict_value in self.observables_array_dict.items():
            if len(observables_array_dict_value.shape) == 1 or len(observables_array_dict_value.shape) == 2:
                observables_array_dict_value[:,timestep] = self.observables_comp_functions_dict[observables_array_dict_key](state, observables_array_dict_value.shape[:-1], observables_array_dict_value.dtype )
            
            elif len(observables_array_dict_value.shape) == 3:
                observables_array_dict_value[:,:,timestep] = self.observables_comp_functions_dict[observables_array_dict_key](state, observables_array_dict_value.shape[:-1], observables_array_dict_value.dtype )

            else:
                raise ValueError('Mattia: class ObservablesDict() working only for rank 1,2,3 arrays')

        if save == True:
            self.save_observables()


    def preprocess_trajectories_before_averaging(self, n_trajectories: list, write_directory: str) -> list :
        
        import os
        os.chdir(write_directory)
        traj_list = []
        #check whether trajectory-directory and first observable in array_dict exist
        for i in range( n_trajectories ):
            if os.path.isdir(str(i))and os.path.isfile(str(i) + '/' + [*self.observables_array_dict][0] ): 
                traj_list.append(i)
        
        #check whether the first observable has been computed for every timestep. if not, remove the trajectory from traj_list    
        counter = 0 
        n_eliminated_traj = 0   
        traj_list_new = traj_list.copy()
        
        for trajectory in traj_list:
            os.chdir(str(trajectory))
            obs1 = np.loadtxt( [*self.observables_array_dict][0] )
            print('obs1[0,-1] in traj {} = {}'.format(trajectory, obs1[0,-1]))
            if  obs1[0,-1] == 0. :
                traj_list_new.pop(counter-n_eliminated_traj)
                n_eliminated_traj += 1
                print('eliminated trajectory ', trajectory )
            
            os.chdir('..')    
            counter += 1
            
        return traj_list_new             
    
    def compute_trajectories_averages_and_errors(self, traj_list: list, read_directory:str, write_directory:str, remove_single_trajectories_results: bool = False ):
        """Given the dictionary 'observables_array_dict' saved in the instance, computes the averages and errors for all observables 
           over all trajectories  and saves them.

        Parameters
        ----------
        traj_list : list
            list of trajectories over which to average
        read_directory : str
            directory in which all trajectories are stored. NOTE: trajectories must be copied here by method "preprocess_trajectories_before_averaging"
        write_directory : str
            directory in which averaged observables and errors will be saved
        remove_single_trajectories_results : bool, optional
            if True,removes all the trajectories in read_directory after having computed the averages, by default False
        """
        import os
        n_trajectories = len(traj_list) #count the number of trajectories which passed the test of "preprocess_trajectories_before_averaging"
        #os.chdir(read_directory) #go to data directory
        # print("I am in: {}".format(os.getcwd() ) )
        # print(os.listdir())
        
        #initialize averaged observables
        averaged_observables_array_dict = self.observables_array_dict.copy()
        for key in averaged_observables_array_dict.copy():
            averaged_observables_array_dict[ key ] = np.zeros( averaged_observables_array_dict[ key ].shape ) #set arrays to zero
            averaged_observables_array_dict[ key + '_av' ] = averaged_observables_array_dict.pop(key) #add '_av' to key names 

        #initialize errors
        stat_errors_observables_array_dict = self.observables_array_dict.copy()            
        for key in stat_errors_observables_array_dict.copy():
            stat_errors_observables_array_dict[ key ] = np.zeros( stat_errors_observables_array_dict[ key ].shape ) #set arrays to zero
            stat_errors_observables_array_dict['err_' + key] = stat_errors_observables_array_dict.pop(key) #add '_av' to key names 
        
        #compute averaged observables
        for trajectory in traj_list: #loop over trajectories
            print('in trajectory {}'.format(trajectory) )
            os.chdir( str( trajectory ) )
            for key in self.observables_array_dict: #loop over observables
                averaged_observables_array_dict[key + '_av'] += np.loadtxt( key ) #FIXME np.load for arrays or rank >=3
            os.chdir('..')        
            
        #normalize
        for key in averaged_observables_array_dict: #loop over observables
                averaged_observables_array_dict[key] /= n_trajectories 
                
        #compute errors    
        for trajectory in traj_list: #loop over trajectories
            os.chdir( str( trajectory ) )  
            for key in self.observables_array_dict: #loop over observables
                obs = np.loadtxt(key) #FIXME np.load for arrays or rank >=3
                obs_av = averaged_observables_array_dict[key + '_av']
                stat_errors_observables_array_dict['err_' + key] =  (obs - obs_av ) ** 2  
            os.chdir('..') 
            
        #normalize errors
        for key in stat_errors_observables_array_dict: 
            stat_errors_observables_array_dict[key] = np.sqrt( stat_errors_observables_array_dict[key] )/n_trajectories
        
        #save observables
        for key in averaged_observables_array_dict:
            with open(key, 'wb') as f:
                np.savetxt(f, averaged_observables_array_dict[key]) #FIXME: works only for real-valued, 1 or 2D arrays
                print('saved data in {}'.format(os.getcwd()))
        
        #save errors
        for key in stat_errors_observables_array_dict:
            with open(key, 'wb') as f:
                np.savetxt(f, stat_errors_observables_array_dict[key]) #FIXME: works only for real-valued, 1 or 2D arrays
            
        #remove single-trajectories folders
        #FIXME: this removes only the trajectories that passed the preprocessing phase!
        if remove_single_trajectories_results:
            import shutil
            for trajectory in traj_list:
                shutil.rmtree(str(trajectory))
               
                
