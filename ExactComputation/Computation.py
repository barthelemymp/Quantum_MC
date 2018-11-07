import numpy as np


class Computation:
    """
    This class allows the theoretical diagonalisation of the Hamiltonian in 
    order to verify our computations.
    The Hamiltonian is 
    """
    
    
    def __init__(self, L, Jx, Jz, periodic = True):
        """
        Let initialize our class
        """
        #length of the chain, pair 
        self.L = 2*L//2
        #Periodicity of the chain
        self.periodic = periodic
        #coupling
        self.Jx = Jx
        self.Jz = Jz
        #matrix
        self.hamiltonian = np.zeros((L, L))
        
        #states
        self.states = np.zeros((2**L, L))
        self.temp = np.array([np.array([k]) for k in range(2**L)], dtype="uint8")
        self.temp = np.unpackbits(self.temp, axis = 1)
        for i in range(2**L):
            for j in range(L):
                num = int(self.temp[i,j])
                self.states[i,j] = 2*num-1
            
    
    
    #For a chain, the hamiltonian can be separated in two halmitonians
    #only acting on pairs. We call it the "even" hamiltonian 
    #and the "odd" hamiltonian. 
    
    def compute_hamiltonian_on_pair(self, pair):
        """
        This method compute the effect of the hamiltonian_two_sites on a pair
        """
        
        
        vec = np.concatenate((pair_to_vec[pair[0]], pair_to_vec[pair[1]]))
        print(vec)
        
        #hamiltonian_two_sites on the pair 
        hamiltoniantwosites = np.zeros((4,4))
        hamiltoniantwosites[0,0] = self.Jz/4
        hamiltoniantwosites[3,3] = self.Jz/4
        hamiltoniantwosites[1,1] = -self.Jz/4
        hamiltoniantwosites[2,2] = -self.Jz/4
        hamiltoniantwosites[1,2] = self.Jx/2
        hamiltoniantwosites[2,1] = self.Jx/2
        
        print(hamiltoniantwosites)

    
    def compute_even_energy(self, i, j):
        """
        We compute the matrix even element between state i and j
        """
        #let reshape i and j
        state_i = self.states[i].reshape(2, self.L//2)
        print(state_i)
        state_j = self.states[j].reshape(self.L//2, 2)
        print(state_j)
        
        return np.trace(np.dot(state_j, state_i))
        
    def set_hamiltonian():
        return
        

c = Computation(8, 1, 1)
#print(c.hamiltonian)
        