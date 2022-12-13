"""One-dimensional boson - spin 1/2 lattice. First make this independently, in order to see what methods and attributes it need.

Later make it inherit from an ABC class 'Lattice'. """
import numpy as np


class BosSpinOneHalfLattice():
    """Vacuum = |1 0 ... 0 >. sigma plus (sp) is the annihilator and sigma minus (sm) is the creator of the spin 1/2 sites,
    b (bm) is the annihilator and b^\dagger is the creator of the boson sites"""
    
    def __init__(self, sites: list, bos_dim: int):
        """Saves sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus, b^\dagger, b, n_bos as instance variables and adds them to the operators dictionary.
        All these operator are saved as instance variables, together with the the identity operator and the vacuum state Vacuum = |1 0 ... 0 > = |up up .... up >.
        sigma plus (sp) is the annihilator and sigma minus (sm) is the creator of the spin 1/2 site
        b (bm) is the annihilatior and b^\dagger (bp) is the creator of the boson site
    
        Parameters
        ----------
        sites : list
            add 0 for boson and 1 for spin 1/2 in the correct order
        bos_dim: int
            local dimension of the boson sites
        """
        # operators
        sx = 0.5 * np.array([[0,1], [1,0]], dtype='complex')
        sy = 0.5 * np.array([[0,-1j], [1j,0]], dtype='complex')
        sz = 0.5 * np.array([[1,0], [0,-1]], dtype='complex')
        sp = sx + 1.j * sy
        sm = sx - 1.j * sy
        bp = np.diag(np.array([np.sqrt(i) for i in range(1, bos_dim)], dtype='complex'), k=-1)
        bm = np.diag(np.array([np.sqrt(i) for i in range(1, bos_dim)], dtype='complex'), k=1)
        n_bos = np.diag(np.array([i+1 for i in range(bos_dim)], dtype='complex'))
        I_spin = np.eye(2, dtype='complex')
        I_bos = np.eye(bos_dim, dtype='complex')
        self.I_spin = I_spin
        self.I_bos = I_bos
        self.sites = sites
        self.n_sites = len(sites)
        operators = {}
        self.operators = operators
        operators.update({'sx': sx})
        operators.update({'sy': sy})
        operators.update({'sz': sz})
        operators.update({'sp': sp})
        operators.update({'sm': sm})
        operators.update({'bp': bp})
        operators.update({'bm': bm})
        operators.update({'n_bos': n_bos})
        # vacuum state
        self.dim_ges = 2**sites.count(1) * bos_dim**sites.count(0)
        vacuum_state = np.zeros(self.dim_ges, dtype='complex')
        vacuum_state[0] = 1.
        self.vacuum_state = vacuum_state
     
    def sso(self, operator_name: str, site: int) -> np.ndarray:
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
            if self.sites[0]:
                single_site_operator = self.I_spin.copy()
            else:
                single_site_operator = self.I_bos.copy()
        
        for i in range(1, self.n_sites):
            if i != site:
                if self.sites[i]:
                    single_site_operator = np.kron(single_site_operator, self.I_spin)
                else:
                    single_site_operator = np.kron(single_site_operator, self.I_bos)
            elif i == site:
                single_site_operator = np.kron(single_site_operator, operator)

        return single_site_operator

