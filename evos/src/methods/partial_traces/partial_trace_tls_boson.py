import numpy as np
def tracing_out_one_tls_from_tls_bosonic_system( which_spin_trace_out: int, rho: np.ndarray, site_types: list, max_bosons: int ) -> np.ndarray:
    """Given a density matrix over tls (spinles fermions or spins) and bosons, traces out one tls.
    'site_types' tells which sites are tls (1) and which are bosons (0). Example: site_types = [1,1,0,1] generates the lattice: tls - tls - bos - tls

    Parameters
    ----------
    which_spin_trace_out : int
        site of spin to be traced out
    rho : np.ndarray
        density matrix of which one spin is to be traced out
    site_types : list
       tells which sites are tls (1) and which are bosons (0)
    max_bosons : int
        bosonic hilbert space dimension - 1
    Returns
    -------
    np.ndarray
        reduced density matrix
    """
    state_zero = np.array( [[ 1, 0] ], dtype='complex' )
    state_one = np.array( [ [0, 1] ], dtype='complex' )
    I_f = np.eye(2, dtype='complex')
    I_b = np.eye(max_bosons + 1, dtype='complex') 
    op_vec_zero_list_left = []
    op_vec_one_list_left = []
    op_vec_zero_list_right = []
    op_vec_one_list_right = []

    #1) determine the vector and operator sting
    for site in range( len(site_types) ):
        
        if site == which_spin_trace_out: 
            op_vec_zero_list_left.append( state_zero )
            op_vec_zero_list_right.append( state_zero.T )
            op_vec_one_list_left.append( state_one )
            op_vec_one_list_right.append( state_one.T )
             
        elif site != which_spin_trace_out and site_types[site] == 1:
            op_vec_zero_list_left.append( I_f )
            op_vec_zero_list_right.append( I_f )
            op_vec_one_list_left.append( I_f )
            op_vec_one_list_right.append( I_f )
        
        elif site != which_spin_trace_out and site_types[site] == 0:
            op_vec_zero_list_left.append( I_b )
            op_vec_zero_list_right.append( I_b )
            op_vec_one_list_left.append( I_b )
            op_vec_one_list_right.append( I_b )    
            
    #2) take the kronecker products
    product_zero_left = op_vec_zero_list_left[0]
    product_zero_right = op_vec_zero_list_right[0]
    product_one_left = op_vec_one_list_left[0]
    product_one_right = op_vec_one_list_right[0] 
       
    for i in range(1, len(site_types)): 
        product_zero_left = np.kron( product_zero_left, op_vec_zero_list_left[i] )
        product_zero_right = np.kron( product_zero_right, op_vec_zero_list_right[i] )
        
        product_one_left = np.kron( product_one_left, op_vec_one_list_left[i] )
        product_one_right = np.kron( product_one_right, op_vec_one_list_right[i] )
        
    #3) compute the partial trace
    rho_reduced = product_zero_left @ rho @ product_zero_right + product_one_left @ rho @ product_one_right
    
    return rho_reduced 
