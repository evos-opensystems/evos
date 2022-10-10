"""Compute Lindblad evolution of system density matrix. This should be, at least in the beginning, the only method that works only with ED and not with MPS"""
import numpy as np
class Lindblad():
    """"""
   
    
    def __init__(self, lindbl_op_list: list, H: np.ndarray, n_sites: int):
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
        rho_v = [] #np.zeros((int(dim_H*(dim_H+1)/2)),dtype='complex')
        for n in range( self.dim_H ):
            for m in range(n, self.dim_H):
                rho_v.append(rho[n,m])
        rho_v = np.array(rho_v, dtype='complex')      
        return rho_v 
    
    def un_vectorize_density_matrix(self, rho_v: np.ndarray) -> np.ndarray :
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
        
    def right_hand_side_lindblad_eq(self, t: int, rho_v: np.ndarray):
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
        """"""
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
        """"""
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
        
        
            
        
        