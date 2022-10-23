"""One-dimensional spin 1/2 lattice. First make this independently, in order to see what methods and attributes it need.
Later make it inherit from an ABC class 'Lattice'. """
import numpy as np

class SpinOneHalfLattice():
    """Vacuum = |1 0 ... 0 > = |up up .... up >. sigma plus (sp) is the annihilator and sigma minus (sm) is the creator"""
    
    def __init__(self, n_sites: int):
        """Saves sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus as instance variables and adds them to the operators dictionary.
        All these operator are saved as instance variables, together with the the identity operator and the vacuum state Vacuum = |1 0 ... 0 > = |up up .... up >.
        sigma plus (sp) is the annihilator and sigma minus (sm) is the creator
        

        Parameters
        ----------
        n_sites : int
            number of spin 1/2 lattice sites
        """
        #operators
        sx = 0.5 * np.array([[0,1], [1,0]],dtype='complex')
        sy = 0.5 * np.array([[0,-1j], [1j,0]],dtype='complex')
        sz = 0.5 * np.array([[1,0], [0,-1]],dtype='complex')
        sp = sx + 1j * sy
        sm = sx - 1j * sy
        I = np.eye(2, dtype='complex')
        self.I = I
        self.n_sites = n_sites 
        operators = {}
        self.operators = operators
        operators.update({'sx':sx})
        operators.update({'sy':sy})
        operators.update({'sz':sz})
        operators.update({'sp':sp})
        operators.update({'sm':sm})
        #vacuum state
        vacuum_state = np.zeros(2**n_sites, dtype='complex')
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
            if i != site:
                single_site_operator = np.kron(single_site_operator,self.I)
            elif i == site:
                single_site_operator = np.kron(single_site_operator,operator)   
    
    #def __mul__(self):
        
            
        return single_site_operator
    
                
        