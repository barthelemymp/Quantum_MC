class Computation():
    """
    This class allows the theoretical diagonalisation of the Hamiltonian in 
    order to verify our computations
    """
    
    def _init_(self, L, Jx, Jz, periodic = True):
        """
        Let initialize our class
        """
        #length of the chain
        self.L = L
        #Periodicity of the chain
        self.periodic = periodic
        #coupling
        self.Jx = Jx
        self.Jy = Jy
        
        #matrix
        self.Hamiltonian = np.array((L, L))
        