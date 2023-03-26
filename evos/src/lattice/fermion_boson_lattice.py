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
        vacuum_state = np.zeros( ( 16 * ( max_bosons + 1 ) ) ** n_sites, dtype='complex')
        vacuum_state[0] = 1
        self.vacuum_state = vacuum_state
           
    
    def sso(self, operator_name: str, site: int, spin: str = '0') -> np.ndarray :
        """Constructs single-site operator. Spinful fermion is described as two TLSs.
         
        Parameters
        ----------
        operator_name : str
            operators defined in init
        site : int
            site on which operator acts
        spin : str, optional
            'up' or 'down' for fermions, by default '0' (bosons)

        Returns
        -------
        np.ndarray
            single site operator acting on whole hilbert space

        Raises
        ------
        ValueError
            check spin input
        """
        
        #get index of operator 
        if spin == 'up':
            operator_site = 2 * site - int( site/4 )
        elif spin == 'down':
            operator_site = 2 * site - int( site/4 ) + 1
        elif spin == '0': #default
            operator_site = (site + 1) * 5 - 1
        elif spin != '0' and spin != 'up' and spin != 'down':
            raise ValueError(" 'spin' must be either 'up' or 'down'. The default is '0' (i.e. boson) ")   

        operator = self.operators[operator_name]
        #initialize single_site_operator
        if site == 0 and spin == 'up':
            single_site_operator = operator.copy()
        else:
            single_site_operator = self.Id_f
        
        for site in range(1, 5 * self.n_sites ):
            if site == operator_site: #operator site
                single_site_operator = np.kron(single_site_operator,operator)
                
            
            if site < operator_site and (site + 1) % 5 != 0: #fermionic site left
                #print('site {} is left fermionic'.format(site))
                single_site_operator = np.kron(single_site_operator,self.Id_f)
            
            if site > operator_site and (site + 1) % 5 != 0: #fermionic site right
                #print('site {} is right fermionic'.format(site))
                single_site_operator = np.kron(single_site_operator,self.parity)    
                
            elif site != operator_site and (site + 1) % 5 == 0: #bosonic site
                #print('site {} is bosonic'.format(site))
                single_site_operator = np.kron(single_site_operator,self.Id_b)
                
        return single_site_operator

                  

             