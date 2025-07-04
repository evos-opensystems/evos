"""One-dimensional lattice for bosons.

First make this independently, in order to see what methods and attributes it need.
Later make it inherit from an ABC class 'Lattice'. """
from typing import List, Union
import numpy as np


class BosonicLatticeMattia():
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
        #operators
        ah = np.zeros((max_bosons+1,max_bosons+1 ) , dtype='complex')
        for i in range(1,max_bosons+1):
            ah[i,i-1] = np.sqrt(i)
        a = np.zeros( ( max_bosons+1, max_bosons+1 ) , dtype='complex' )
        for i in range(max_bosons):
            a[i,i+1] = np.sqrt(i+1)
        
        
        I = np.eye( max_bosons + 1, dtype='complex')
        
        self.n_sites = n_sites 
        self.max_bosons = max_bosons
        self.I = I

        operators = {}
        self.operators = operators
        operators.update( { 'ah' :ah } )
        operators.update( { 'a' :a } )
        
        #vacuum state
        vacuum_state = np.zeros( ( max_bosons + 1 )**n_sites, dtype='complex')
        vacuum_state[0] = 1
        self.vacuum_state = vacuum_state

    def get_fock_state(self, occupations: Union[List[int], np.ndarray]) -> np.ndarray:
        """Given occupations on each site, generate the occupation.

        Function uses the python built-in `int(str, base)`, which is limited to 2-36 for the local dimension. 
        See https://docs.python.org/3/library/functions.html#int

        Args:
            occupations (Union[List[int], np.ndarray]): 1D list of occupations corresponding to the site

        Returns:
            scipy.sparse.csr_matrix: Fock state
        """

        assert len(occupations) == self.n_sites, f"The number of sites in the input occupation list ({len(occupations) }) does not match the number of sites in the lattice ({self.n_sites})."
        assert np.all([0 <= occupation <= self.max_bosons for occupation in occupations]), f"Occupations must be between 0 and max_bosons = {self.max_bosons}."

        if self.max_bosons > 35:
            raise NotImplementedError(f"Local dimension = {self.max_bosons + 1} is too large for the built-in int(str, base) function.")
            # If this happens, and you need it, implement the conversion manually
            # idx = sum_i(occupation_i * (max_bosons + 1)**i) from LSB to MSB

        if self.max_bosons > 9:
            # Since anything more than 9/site i.e. localdim = 10 is not representable by a single digit
            # convert occupations to letters
            occupations: List[Union[str, int]] = [chr((_occ-10) + 65) if _occ > 9 else _occ for _occ in occupations]
        
        _occ_str = "".join([str(occupation) for occupation in occupations])
        _occ_idx = int(_occ_str, self.max_bosons + 1) # str in base (max_bosons + 1) to int

        _state = np.zeros(self.vacuum_state.shape, dtype='complex')
        _state[_occ_idx] = 1

        return _state
    
    def sso(self, operator_name: str, site: int) -> np.ndarray :
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
                single_site_operator = np.kron(single_site_operator,operator)  
            elif i != site:
                single_site_operator = np.kron(single_site_operator,self.I)                   
            
        return single_site_operator 