import numpy as np

class DotWithOscillatorLattice():
    """"Creates the following lattice: fermion - fermion - boson.
        The fermions are spinless
    """
    def __init__(self, max_bosons: int ):
        """Saves single site operators as instance variables and adds them to the operators dictionary.
        All these operator are saved as instance variables, together with the the identity operator and the vacuum state. Vacuum = |1 0 ... 0 > = |up up .... up >.
        
        Parameters
        ----------
        max_bosons : int
            local dimension on bosonic sites - 1    
        """
        #operators
        ch = np.array( [ [0,0], [1,0] ], dtype='complex' )
        c = np.array( [ [0,1], [0,0] ], dtype='complex' )
        
        ah = np.zeros( (max_bosons + 1,max_bosons + 1), dtype='complex'  )
        for i in range( 1, max_bosons + 1 ):
            ah[i, i - 1] = np.sqrt(i)
            
        a = np.zeros( ( max_bosons + 1, max_bosons + 1), dtype='complex'  )
        for i in range(max_bosons):
            a[i, i + 1 ] = np.sqrt(i+1)
         
        self.parity = np.array( [ [1,0], [0,-1] ], dtype='complex' )
        self.Id_f = np.eye(2, dtype='complex')
        self.Id_b = np.eye(max_bosons+1, dtype='complex')        
        operators = {}
        self.operators = operators
        operators.update({'ch':ch})
        operators.update({'c':c})
        operators.update({'ah':ah})
        operators.update({'a':a})
        
        #vacuum state
        vacuum_state = np.zeros( ( 8 * (max_bosons + 1) ), dtype='complex')
        vacuum_state[0] = 1
        self.vacuum_state = vacuum_state
           
    
    def sso(self, operator_name: str, site: int) -> np.ndarray :
        """Constructs single-site operator. Spinful fermion is described as two TLSs.
         
        Parameters
        ----------
        operator_name : str
            operators defined in init
        site : int
            site on which operator acts
        
        Returns
        -------
        np.ndarray
            single site operator acting on whole hilbert space

        Raises
        ------
        ValueError
            check spin input
        """
       
        operator = self.operators[operator_name]
        #hardcoding the 4 possibilities
        if site == 0:
            single_site_operator = np.kron( operator.copy(), np.kron( self.parity, self.Id_b ) )
        
        elif site == 1:
            single_site_operator = np.kron( self.Id_f, np.kron( operator.copy(), self.Id_b ) )
            
        elif site == 2:
            single_site_operator = np.kron( self.Id_f, np.kron( self.Id_f, operator.copy() ) )    

        return single_site_operator    