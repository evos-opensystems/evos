import numpy as np

class FermionBosonLattice():
    """"Creates the following lattice: fermion - fermion - boson - fermion - fermion - boson ...
        The fermions are spinful
    """
    def __init__(self, n_sites: int, max_bosons: int ):
        """Saves single site operators as instance variables and adds them to the operators dictionary.
        All these operator are saved as instance variables, together with the the identity operator and the vacuum state Vacuum = |1 0 ... 0 > = |up up .... up >.
        
        Parameters
        ----------
        n_sites : int
            number of supersites. one supersite is 'fermion - fermion - boson'
        max_bosons : int
            local dimension on bosonic sites - 1    
        """
        #operators
        ch = np.array( [ [0,0], [1,0] ], dtype='complex' )
        c = np.array( [ [0,1], [0,0] ], dtype='complex' )
        ah = ch.copy()
        a = c.copy()
        self.parity = np.array( [ [1,0], [0,-1] ], dtype='complex' )
        self.Id_f = np.eye(2)
        self.Id_b = np.eye(max_bosons+1)        
        self.n_sites = n_sites 
        operators = {}
        self.operators = operators
        operators.update({'ch':ch})
        operators.update({'c':c})
        operators.update({'ah':ah})
        operators.update({'a':a})
        
        #vacuum state
        vacuum_state = np.zeros( ( 4 * ( max_bosons + 1 ) ) ** n_sites, dtype='complex')
        vacuum_state[0] = 1
        self.vacuum_state = vacuum_state


    def sso_fer(self, operator_name: str, spin: str, site: int) -> np.ndarray :
        """Construct bosonic single-site operator. Spinful fermion is described as two TLSs. even sites: up, odd sites: down"""
                
        operator = self.operators[operator_name]
        if site == 0 and spin:
            single_site_operator = operator.copy()
        else:
            single_site_operator = self.Id_f.copy()
        
        for i in range(1, self.n_sites):  
            if i < site:
                single_site_operator = np.kron(single_site_operator, self.Id_f)
                single_site_operator = np.kron(single_site_operator, self.Id_f)
                single_site_operator = np.kron(single_site_operator, self.Id_b)
                
            elif i > site:
                single_site_operator = np.kron(single_site_operator,self.parity)                    
                single_site_operator = np.kron(single_site_operator,self.parity) 
                single_site_operator = np.kron(single_site_operator,self.Id_b) 
                
            elif i == site and spin == 'up':
                single_site_operator = np.kron(single_site_operator,operator)  
                single_site_operator = np.kron(single_site_operator,self.Id_f) #FIXME: correct????
                single_site_operator = np.kron(single_site_operator,self.Id_b)
            
            elif i == site and spin == 'down':
                single_site_operator = np.kron(single_site_operator,self.Id_f)
                single_site_operator = np.kron(single_site_operator,operator)  
                single_site_operator = np.kron(single_site_operator,self.Id_b)               
            
        return single_site_operator

                  

             