class Lattice():
    """Choose a representation and a lattice file to be imported.
    """

    def __init__(self, representation: str):  #use typing module to define new types
        """Initialize the object by storing the representation time as exact diagonalization (ed) or matrix product state (mps)

        Parameters
        ----------
        representation : str
            must be either 'mps' or 'ed'
        """
        if representation == 'mps':
            import evos.src.representation.mps as mps
            self.mps = mps
            
        elif representation == 'ed':
            import evos.src.representation.ed as ed 
            self.ed = ed
            
        #else: #ADD AN ERROR THAT IS RAISED ONLY WHEN INPUT IS GIVEN BUT WRONG, IN ORDER NOT TO OVERWRITE THE TypeError
            #raise IOError('the valid representations are: {0}'.format(representation_list))    
            #arguments types contained in lat.Lattice.__init__.__annotations__
            
    def specify_lattice(self, lattice_name: str):
        """Import the selected lattice if it has been implemented. Need to add a NotImplementedError!

        Parameters
        ----------
        lattice_name : str
            name of the lattice file
        """
        if lattice_name == 'spin_one_half_lattice':
            import evos.src.lattice.spin_one_half_lattice as spin_one_half_lattice
            self.spin_one_half_lattice = spin_one_half_lattice

        if lattice_name == 'spinful_fermions_lattice':
            import evos.src.examples.spinful_fermions_lattice as spinful_fermions_lattice
            self.spinful_fermions_lattice = spinful_fermions_lattice
 
        
