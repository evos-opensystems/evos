"""
Copy of bosonic_lattice_mattia, but with sparse matrices
Made by Yudong Sun, 2025
"""

"""One-dimensional lattice for bosons.

First make this independently, in order to see what methods and attributes it need.
Later make it inherit from an ABC class 'Lattice'. """
import numpy as np
import scipy.sparse
from typing import List, Union

class BosonicLatticeSparse():
    """Vacuum = |1 0 ... 0 > = zero particles present."""
    
    def __init__(self, n_sites: int, max_bosons: int):
        """Saves a, ah, I as instance variables and adds them to the operators dictionary.
        All these operator are saved as instance variables, together with the the identity operator and the vacuum state Vacuum = |1 0 ... 0 > = |up up .... up >.
        sigma plus (sp) is the annihilator and sigma minus (sm) is the creator
        

        Parameters
        ----------
        n_sites : int
            number of bosonic lattice sites
        max_bosons : int
            maximal number of bosons allowed per site, i.e. local Hilbert space dimension -1     
        """
        # Operators using sparse matrices (csr_matrix)
        ah_row_idx = np.arange(1, max_bosons + 1)
        ah_col_idx = ah_row_idx - 1
        ah_data    = np.sqrt(ah_row_idx)

        ah = scipy.sparse.csr_matrix((ah_data, (ah_row_idx, ah_col_idx)), shape = (max_bosons+1, max_bosons+1), dtype='complex')

        # ah = np.zeros((max_bosons+1,max_bosons+1 ) , dtype='complex')
        # for i in range(1,max_bosons+1):
        #     ah[i,i-1] = np.sqrt(i)

        a_row_idx = np.arange(0, max_bosons)
        a_col_idx = a_row_idx + 1
        a_data    = np.sqrt(a_col_idx)

        a = scipy.sparse.csr_matrix((a_data, (a_row_idx, a_col_idx)), shape = (max_bosons+1, max_bosons+1), dtype='complex')

        # a = np.zeros( ( max_bosons+1, max_bosons+1 ) , dtype='complex' )
        # for i in range(max_bosons):
        #     a[i,i+1] = np.sqrt(i+1)
        
        I = scipy.sparse.eye( max_bosons + 1, dtype='complex', format = "csr")
        
        self.n_sites = n_sites
        self.max_bosons = max_bosons
        self.I = I

        operators = {}
        self.operators = operators
        operators.update( { 'ah' : ah } )
        operators.update( { 'a'  : a  } )
        
        #vacuum state
        vac_row  = [0]
        vac_col  = [0]
        vac_data = [1]
        vacuum_state = scipy.sparse.csr_matrix((vac_data, (vac_row, vac_col)), shape = ((max_bosons + 1)**n_sites, 1), dtype='complex')
        # vacuum_state[0] = 1
        self.vacuum_state = vacuum_state

    def get_fock_state(self, occupations: Union[List[int], np.ndarray]) -> scipy.sparse.csr_matrix:
        """Given occupations on each site, generate the occupation

        Args:
            occupations (Union[List[int], np.ndarray]): 1D list of occupations corresponding to the site

        Returns:
            scipy.sparse.csr_matrix: Fock state
        """

        assert len(occupations) == self.n_sites, f"The number of sites in the input occupation list ({len(occupations) }) does not match the number of sites in the lattice ({self.n_sites})."
        assert np.all(0 <= occupation <= self.max_bosons for occupation in occupations), f"Occupations must be between 0 and max_bosons = {self.max_bosons}."
        
        _occ_str = "".join([str(occupation) for occupation in occupations])
        _occ_idx = int(_occ_str, self.max_bosons + 1) # str in base (max_bosons + 1) to int

        _state_row  = [_occ_idx]
        _state_col  = [0]
        _state_data = [1]
        _state = scipy.sparse.csr_matrix((_state_data, (_state_row, _state_col)), shape = self.vacuum_state.shape, dtype='complex')

        return _state
    
    def sso(self, operator_name: str, site: int) -> scipy.sparse.csr_matrix :
        """Given an operator name and a site number, it computes the single site operator acting on the whole Hilbert space
        by computing kronecker products with the identity left and right to the site on which the operator is applied.

        Parameters
        ----------
        operator_name : str
            key of the dictionary instance variable 'operators' computed by the __init__() method
        site : int
            lattice site on wich the single site operator acts

        Returns
        -------
        np.ndarray
            single site operator acting on the whole Hilbert space
        """
        operator = self.operators[operator_name]
        if site == 0:
            single_site_operator = operator.copy()
        elif site != 0:
            single_site_operator = self.I.copy()
        
        for i in range(1, self.n_sites):               
            if i == site:
                single_site_operator = scipy.sparse.kron(single_site_operator, operator, format = "csr")  
            elif i != site:
                single_site_operator = scipy.sparse.kron(single_site_operator, self.I, format = "csr")
            
        return single_site_operator 