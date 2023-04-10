import numpy as np


def tracing_out_one_spin_out_of_n( which_spin_trace_out: int, rho: np.ndarray, n_spins: int ) -> np.ndarray:
    """Given a density matrix over 'n_spins' spin sites, traces out one spin.

    Parameters
    ----------
    which_spin_trace_out : int
        site of spin to be traced out
    rho : np.ndarray
        density matrix of which one spin is to be traced out
    n_spins : int
       number of spin sites in rho

    Returns
    -------
    np.ndarray
        reduced density matrix
    """
    state_zero = np.array( [[ 1, 0] ], dtype='complex' )
    state_one = np.array( [ [0, 1] ], dtype='complex' )
    I = np.eye(2)
        
    op_vec_zero_list_left = []
    op_vec_one_list_left = []
    op_vec_zero_list_right = []
    op_vec_one_list_right = []

    #1) determine the vector and operator sting
    for i in range(n_spins):
        
        if i == which_spin_trace_out: 
            op_vec_zero_list_left.append( state_zero )
            op_vec_zero_list_right.append( state_zero.T )
            op_vec_one_list_left.append( state_one )
            op_vec_one_list_right.append( state_one.T )
             
        elif i != which_spin_trace_out:
            op_vec_zero_list_left.append( I )
            op_vec_zero_list_right.append( I )
            op_vec_one_list_left.append( I )
            op_vec_one_list_right.append( I )
            
    #2) take the kronecker products
    product_zero_left = op_vec_zero_list_left[0]
    product_zero_right = op_vec_zero_list_right[0]
    product_one_left = op_vec_one_list_left[0]
    product_one_right = op_vec_one_list_right[0] 
       
    for i in range(1,n_spins): 
        product_zero_left = np.kron( product_zero_left, op_vec_zero_list_left[i] )
        product_zero_right = np.kron( product_zero_right, op_vec_zero_list_right[i] )
        
        product_one_left = np.kron( product_one_left, op_vec_one_list_left[i] )
        product_one_right = np.kron( product_one_right, op_vec_one_list_right[i] )
        
    #3) compute the partial trace
    rho_reduced = product_zero_left @ rho @ product_zero_right + product_one_left @ rho @ product_one_right
    
    return rho_reduced 


def trace_out_boson_from_one_spin_one_boson_system( rho: np.ndarray, max_bosons: int ) -> np.ndarray: 
    
    """Given a density matrix over 'n_spins' spin sites, traces out one spin. Spin left and boson right.

    Parameters
    ----------
    rho : np.ndarray
        density matrix for one spin and one boosn of which one boson is to be traced out
    max_bosons : int
        local hilbert space dimension - 1 for the bosonic site

    Returns
    -------
    np.ndarray
        reduced density matrix
    """
    Is = np.eye(2, dtype='complex') #identity for spin sites
    
    # 1) make a dictionary with all the bosonic basis states
    bos_states_dict = {}
    zeros =  np.zeros( ( 1, max_bosons + 1 ), dtype='complex' )
    for i in range(max_bosons + 1):
        state = zeros.copy()
        state[0,i] = 1.
        bos_states_dict.update( {'bos_state_' + str(i) : state} )
        
    #2) compute the partial trace
    rho_reduced = np.zeros( (2,2), dtype='complex' )
    for i in range( len(bos_states_dict) ):
        rho_reduced += np.kron( Is, bos_states_dict[ 'bos_state_' + str(i) ] ) @  rho @ np.kron( Is, bos_states_dict[ 'bos_state_' + str(i) ].T )
    
    return rho_reduced   