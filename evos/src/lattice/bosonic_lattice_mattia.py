"""One-dimensional lattice for bosons.

First make this independently, in order to see what methods and attributes it need.
Later make it inherit from an ABC class 'Lattice'. """
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