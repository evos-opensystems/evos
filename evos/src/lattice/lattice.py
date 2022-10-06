class Lattice():
    """Generates lattice for choosen particles species and representation"""
    
    def __init__(self, representation:str):  #typing module to define types
        #representation_list = ['mps', 'ed']
        if representation == 'mps':
            import evos.src.representation.mps as mps
            self.mps = mps
            
        elif representation == 'ed':
            import evos.src.representation.ed as ed 
            self.ed = ed
            
        #else: #ADD AN ERROR THAT IS RAISED ONLY WHEN INPUT IS GIVEN BUT WRONG, IN ORDER NOT TO OVERWRITE THE TypeError
            #raise IOError('the valid representations are: {0}'.format(representation_list))    
            #arguments types contained in lat.Lattice.__init__.__annotations__