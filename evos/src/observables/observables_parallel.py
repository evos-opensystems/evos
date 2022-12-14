import numpy as np

class ObservablesDict():
    def __init__(self):
        """Initialize two dictionaries, 'observables_array_dict' in which the computed observables are stored and 
        'observables_comp_functions_dict' in which the function that is used to compute the observable at each timestep is stored.
        They are both saved in the instance.
        """
        self.observables_array_dict = {}
        self.observables_comp_functions_dict = {}

    def initialize_observable(self, obs_name: str, obs_array_shape: tuple, n_timesteps: int, dtype='float'):
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
        obs_array_shape = sum((obs_array_shape, (n_timesteps+1,)), ()) 
        obs_array = np.zeros(obs_array_shape, dtype=dtype)

        self.observables_array_dict.update({obs_name:obs_array})

    def save_observables(self, trajectory: int):
        """Saves all the observables contained in the dictionary 'observables_array_dict'.
        NOTE: in numpy arrays of rank > 2 need to be pickled.
        """

        for key, value in self.observables_array_dict.items():
            #print(key, value)
            if len(value.shape) < 3:
                with open(key+'_'+str(trajectory), 'wb') as f:
                    np.savetxt(f, value)
            if len(value.shape) >= 3:
                with open(key+'_'+str(trajectory), 'wb') as f:
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
        self.observables_comp_functions_dict.update({obs_name:observable_computing_function})


    def compute_all_observables_at_one_timestep(self, state, timestep: int, trajectory: int, save=True):
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
            self.save_observables(trajectory)

