"""One-dimensional lattice for spinless fermions.

First make this independently, in order to see what methods and attributes it need.
Later make it inherit from an ABC class 'Lattice'. """
import numpy as np

class SpinlessFermionsLattice():
    """Vacuum = |1 0 ... 0 > = zero particles present."""
    
    def __init__(self, n_sites: int):
        """Saves ch, c, parity, I as instance variables and adds them to the operators dictionary.
        All these operator are saved as instance variables, together with the the identity operator and the vacuum state Vacuum = |1 0 ... 0 > = |up up .... up >.
        
        Parameters
        ----------
        n_sites : int
            number of spin 1/2 lattice sites
        """
        #operators
        ch = np.array( [ [0,0], [1,0] ], dtype='complex' )
        c = np.array( [ [0,1], [0,0] ], dtype='complex' )
        parity = np.array( [ [1,0], [0,-1] ], dtype='complex' )
        I = np.eye(2, dtype='complex')
        self.parity = parity
        self.I = I
        self.n_sites = n_sites 
        operators = {}
        self.operators = operators
        operators.update({'ch':ch})
        operators.update({'c':c})
        
        #vacuum state
        vacuum_state = np.zeros(2**n_sites, dtype='complex')
        vacuum_state[0] = 1
        self.vacuum_state = vacuum_state
     
    def sso(self, operator_name: str, site: int) -> np.ndarray :
        """Given an operator name and a site number, it computes the single site operator acting on the whole Hilbert space
        by computing kronecker products with the identity left and parities right to the site on which the operator is applied.

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
            if i < site:
                single_site_operator = np.kron(single_site_operator,self.I)
            elif i > site:
                single_site_operator = np.kron(single_site_operator,self.parity)                    
            elif i == site:
                single_site_operator = np.kron(single_site_operator,operator)               
            
        return single_site_operator
    
                
        