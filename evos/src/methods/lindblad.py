"""Compute Lindblad evolution of system density matrix. This should be, at least in the beginning, the only method that works only with ED and not with MPS"""
import numpy as np

class Lindblad():
    """Compute the Lindblad time-evolution of a density matrix and the desired scalar observables (only densities, no correlation funcitions).
    Should be compatible with every lattice, but only with the 'ed' and not the 'mps' representation, at least for the time being.
    """
   
    
    def __init__(self, lindbl_op_list: list, H: np.ndarray, n_sites: int):
        """initialize the Lindblad() object by computing the complex conjugate of the Lindblad operators and setting some import

        Parameters
        ----------
        lindbl_op_list : list
            list containing the lindblad operators
        H : np.ndarray
            Hamiltonian matrix
        n_sites : int
            number of sites of the one-dimensional lattice

        Raises
        ------
        TypeError
            check that all entries of lindbl_op_list are of type np.matrix and not np.ndarray in order to be able to hermitian conjugate them
        """
        
        if not all( type(op) == np.matrix for op in lindbl_op_list ):
            raise TypeError('please cast all lindblad operators into numpy matrices in order to enable the use of the numpy .H method')
        lindbl_op_list_conj = []
        for op in lindbl_op_list:
            lindbl_op_list_conj.append( op.H)
        self.lindbl_op_list = lindbl_op_list
        self.lindbl_op_list_conj = lindbl_op_list_conj
        
        self.H = H
        self.n_sites = n_sites
        self.dim_H = H.shape[0]   
    
    #def lindblad_time_evolution(self, rho_0, t_max, dt):
        """"""
    def ket_to_projector(self, ket: np.ndarray) -> np.ndarray :
        return np.outer(np.conjugate(ket), ket)   
    
    def vectorize_density_matrix(self, rho: np.ndarray) -> np.ndarray :
        """Takes a density operator in matrix form and writes the upper triangular part including the diagonal of it into a vector.
        The lower triangular part is excluded because density matrices are hermitian.

        Parameters
        ----------
        rho : np.ndarray
            input density operator in matrix form

        Returns
        -------
        np.ndarray
            vector containing the upper triangular part of the density matrix including the diagonal
        """
        rho_v = [] #np.zeros((int(dim_H*(dim_H+1)/2)),dtype='complex')
        for n in range( self.dim_H ):
            for m in range(n, self.dim_H):
                rho_v.append(rho[n,m])
        rho_v = np.array(rho_v, dtype='complex')      
        return rho_v 
    
    def un_vectorize_density_matrix(self, rho_v: np.ndarray) -> np.ndarray :
        """Takes a vector containing the upper triangular part of a density operator including the diagonal and turns it into a matrix.
        First, the upper triangular part is reshaped into a matrix. Then it is mirrored and complex conjugated to fill up the lower triangular part.

        Parameters
        ----------
        rho_v : np.ndarray
                vector containing the upper triangular part of a density operator including the diagonal
            

        Returns
        -------
        np.ndarray
            density operator in form of a matrix
        """
        rho = np.zeros( (self.dim_H, self.dim_H ),dtype="complex" )
        #compute upper triangular part
        rho_v_count=0
        for n in range(self.dim_H):
            for  m in range(n,self.dim_H):
                rho[n,m] = rho_v[rho_v_count]
                rho_v_count+=1
        #mirror and complex conjugate upper triangular part
        for n in range(self.dim_H):
            for  m in range(0,n):
                rho[n,m] = np.conjugate(rho[m,n])
        return rho
    #def lindblad_time_evolution(self, rho_0, t_max, dt):
        
    def right_hand_side_lindblad_eq(self, t: int, rho_v: np.ndarray) -> np.ndarray:
        """Computes the right hand side of the lindblad equation from a vectorized density matrix and returns it as a vectorized matrix 

        Parameters
        ----------
        t : int
            always set it to zero
        rho_v : np.ndarray
            vectorized upper triangular part of density matrix

        Returns
        -------
        np.ndarray
            vectorized right hand side of the lindblad matrix equation expression
        """
        #arrange vectorized density matrix into matrix
        rho = self.un_vectorize_density_matrix(rho_v)
        #compute rhs of Lindblad equation
        rhs = -1j*np.matmul(self.H,rho) +1j*np.matmul(rho, self.H)
        for i in range( len(self.lindbl_op_list) ):
            rhs += -0.5 * np.matmul( self.lindbl_op_list_conj[i] , np.matmul( self.lindbl_op_list[i], rho) ) - 0.5 * np.matmul( rho, np.matmul (self.lindbl_op_list_conj[i], self.lindbl_op_list[i]) ) + np.matmul( self.lindbl_op_list[i], np.matmul(rho, self.lindbl_op_list_conj[i]))
        #revectorize density matrix
        rhs_v = self.vectorize_density_matrix(rhs)
        return rhs_v
        
    def solve_lindblad_equation(self, rho_0: np.ndarray, dt: float, t_max: float) -> np.ndarray:
        """Solves the Lindblad equation for a given system, initial state and time interval.
        The Hamiltonian and the list of lindblad operators are taken from the instance and thus passed as input arguments.

        Parameters
        ----------
        rho_0 : np.ndarray
            density operator for initial state in matrix form
        dt : float
            timestep
        t_max : float
            maximal evolution time

        Returns
        -------
        np.ndarray
            rank 3 tensor with rho_res[:,:,time] being the density matrix at the timestep 'time'
        """
        from scipy.integrate import solve_ivp
        
        rho_0_v =  self.vectorize_density_matrix(rho_0)
        n_timesteps = int(t_max/dt)
        self.n_timesteps = n_timesteps
        time_v = np.linspace(0, t_max, n_timesteps )
        res = solve_ivp(self.right_hand_side_lindblad_eq, (0,t_max), rho_0_v,t_eval=time_v)
        
        #rearrange the components of the matrix res (=vector with rho_res components at every timestep) into a rank3 tensor (density matrix at every timestep)
        rho_res = np.zeros( ( self.dim_H, self.dim_H, n_timesteps),dtype='complex')
        rho_v_count=0
        for n in range(self.dim_H):
            for  m in range(n,self.dim_H):
                rho_res[n,m,:] = res.y[rho_v_count,:]
                rho_v_count+=1
            
        for n in range(self.dim_H):
            for  m in range(0,n):
                rho_res[n,m,:] = np.conjugate(rho_res[m,n])
        
        return rho_res
        
    def compute_observables(self,rho_res: np.ndarray, names_and_operators_dict: dict, dt: float, t_max: float ) -> dict:
        """Preliminary version of method used to compute observables. For now it is limited to compute scalar expectation values at each
        timestep, as for instance densities. Needs to be updated to be able to compute correlation functions.

        Parameters
        ----------
        rho_res : np.ndarray
            rank 3 tensor containing density matrix at each timestep, i.e. the return of solve_lindblad_equation()
        names_and_operators_dict : dict
            a dictionary whose keys are the names to be given to the observables and whose values are the operators of which the expectation
            values need to be computed
        dt : float
            timestep
        t_max : float
            maximal simulation time

        Returns
        -------
        dict
            Dictionary whose keys are the the same as the keys of the input 'names_and_operators_dict' and whose values are numpy vectors 
            storing the expectation values for the relative operator at each timestep.
        """
        obs_vector = np.zeros(self.n_timesteps)
        #initialize numpy vectors to save observables
        observables_dict = {}
        for key in names_and_operators_dict:
            observables_dict.update( {key : obs_vector.copy()} )
        
        #compute observables
        for i in range(self.n_timesteps):
            
            for key in observables_dict:
                observables_dict[key][i] = np.trace( np.matmul( names_and_operators_dict[key], rho_res[:,:,i] ) )
        return observables_dict
        
        
            
        
        