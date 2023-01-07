"""One-dimensional lattice alternating a spin1/2 and a bosonic site.
For example, with 'n_sites=2' one generates the lattice 'spin1/2 - boson - spin1/2 boson '

First make this independently, in order to see what methods and attributes it need.
Later make it inherit from an ABC class 'Lattice'. """
import numpy as np

class BosonicLatticeMattia():
    """Vacuum = |1 0 ... 0 > = zero particles present."""
    
    def __init__(self, n_sites: int, max_bosons: int):
        """Saves a, ah, sx, sy, sz, I as instance variables and adds them to the operators dictionary.
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
        #spin
        sx = 0.5 * np.array([[0,1], [1,0]],dtype='complex')
        sy = 0.5 * np.array([[0,-1j], [1j,0]],dtype='complex')
        sz = 0.5 * np.array([[1,0], [0,-1]],dtype='complex')
        sp = sx + 1j * sy
        sm = sx - 1j * sy
        Is = np.eye(2, dtype='complex')

        #boson
        ah = np.zeros((max_bosons+1,max_bosons+1 ) , dtype='complex')
        for i in range(1,max_bosons+1):
            ah[i,i-1] = np.sqrt(i)
        a = np.zeros( ( max_bosons+1, max_bosons+1 ) , dtype='complex' )
        for i in range(max_bosons):
            a[i,i+1] = np.sqrt(i+1)
        Ib = np.eye( max_bosons + 1, dtype='complex')
        
        self.n_sites = n_sites 
        self.max_bosons = max_bosons
        self.Is = Is
        self.Ib = Ib

        operators = {}
        self.operators = operators
        operators.update({'sx':sx})
        operators.update({'sy':sy})
        operators.update({'sz':sz})
        operators.update({'sp':sp})
        operators.update({'sm':sm})
        operators.update( { 'ah' :ah } )
        operators.update( { 'a' :a } )
        
        #vacuum state
        vacuum_state = np.zeros( 2 ** n_sites * ( max_bosons + 1 )**n_sites, dtype='complex')
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
        if operator_name == a or operator_name == ah:
            spin_or_bos = 'bos'
            site = 2 * site + 1
            
        else:
            spin_or_bos = 'spin'
            site = 2 * site
            
        #NOTE:all even sites are spin sites, all odd ones ar bosonic sites
        
        operator = self.operators[operator_name]
        #3 cases: Is x Ib, op x Ib, Is x op, 
        if site == 0: # op x Ib
            single_site_operator = np.kron( operator.copy(), self.Ib )
        
        elif site == 1: #Is x op
            single_site_operator = np.kron( self.Is, operator.copy() )    
            
        elif site > 1: #Is x Ib
            single_site_operator = np.kron( self.Is.copy(), self.Ib.copy() )
        
        for i in range(2, 2 * self.n_sites):
            #3 cases: Is x Ib, op x Ib, Is x op
            if i == site and i % 2 == 0: # op x Ib
                single_site_operator = np.kron( single_site_operator, np.kron( operator.copy(), self.Ib ) )
            
            elif i == site and i % 2 == 1: #Is x op
                single_site_operator = np.kron( single_site_operator, np.kron( self.Is, operator.copy() ) )   
                
            elif != site: #Is x Ib
                single_site_operator = np.kron( single_site_operator, np.kron( self.Is.copy(), self.Ib.copy() ) )
                
        return single_site_operator        
        
        
        
        