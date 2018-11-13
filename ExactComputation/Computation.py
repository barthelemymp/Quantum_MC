# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:18:03 2018

@author: etiennecamphuis
"""

import numpy as np


class Chain:
    """
    This class allows the theoretical diagonalisation of the Hamiltonian in 
    order to verify our computations.
    
    L must be less than 8
    """
    
    
    def __init__(self, L, Jx, Jz, periodic = True):
        """
        Let initialize our chain
        """
        #length of the chain, pair 
        self.L = 2*L//2
        #Periodicity of the chain
        self.periodic = periodic
        #coupling
        self.Jx = Jx
        self.Jz = Jz
        #matrix
        self.hamiltonian = np.zeros((2**self.L, 2**self.L))
        
        #states
        self.states = np.zeros((2**self.L, 8))
        
        #we use the binary expression of integer from 1 to 2**L to create the different spin states
        self.temp = np.array([np.array([k]) for k in range(2**self.L)], dtype="uint8")
        self.temp = np.unpackbits(self.temp, axis = 1)
        for i in range(2**L):
            for j in range(8):
                num = int(self.temp[i,j])
                self.states[i,j] = 2*num-1
        
        #at least
        self.states = self.states[:, 8-self.L:8]
        
        #energies of eigen states
        self.energies = np.array([])
    
    
    
    #For a chain, the hamiltonian can be separated in halmitonians
    #only acting on pairs.
    
    def compute_hamiltonian_on_pair(self, pair1, pair2):
        """
        This method compute < pair 2 | hamiltonian_two_sites | pair 1 >
        """
        #instanciate
        upup = np.array([1.,1.])
        downdown = - upup
        updown = np.array([1.,-1.])
        downup = - updown
        #if up - up, the element matrix is non zero only for pair2 = up - up
        if np.array_equal(pair1, upup):
            return self.Jz/4*np.array_equal(pair2, upup)
        #if down - down
        elif np.array_equal(pair1, downdown):
            return self.Jz/4*np.array_equal(pair2, downdown)
        #if up - down, either pair2 is down - up or up -down
        elif np.array_equal(pair1, updown):
            return -self.Jz/4*np.array_equal(pair2, updown) + self.Jx/2*np.array_equal(pair2, downup)
       #if down - up
        return -self.Jz/4*np.array_equal(pair2, downup) + self.Jx/2*np.array_equal(pair2, updown)
    

    def compute_element_matrix(self, i, j):
        """
        This method compute the matrix even element between state i and j using
        self.compute_hamiltonian_on_pair()
        """
        #let get i and j
        state_i = self.states[i]
        state_j = self.states[j]
        
        
        #result
        sum = 0
        
        
        #let compute the pair to pair hamiltonian
        for k in range(0, self.L-1):
            sum += self.compute_hamiltonian_on_pair(state_i[k:(k+2)], state_j[k:(k+2)])
               
        
        #if the chain is periodic
        if self.periodic:
            sum += self.compute_hamiltonian_on_pair(np.array([state_i[self.L-1], state_i[0]]),
                                                    np.array([state_j[self.L-1], state_j[0]]))

        return sum
        
        
    def set_hamiltonian(self):
        """
        we initialize the hamiltonian using the methods defined above
        """
        for i in range(2**self.L):
            for j in range(i, 2**self.L):
                temp = self.compute_element_matrix(i,j)
                self.hamiltonian[i,j] = temp
                self.hamiltonian[j,i] = temp
    
    def set_eigenvalues(self):
        
        self.set_hamiltonian()
        
        """
        The library np.linalg allows us to compute the eigenvalues and thus the energies 
        of the eigenstates of the hamiltonian
        """
        self.energies = np.linalg.eigvals(self.hamiltonian)
    
    def get_fundamental(self):
        self.set_eigenvalues()
        return np.min(self.energies)



def compute_fundamental_chain(L, Jx, Jz, s = 'result_exact_computation.txt', periodic = True):
    with open(s, 'w') as fichier:
        fichier.write("The fundamental's energy for a " + periodic*"periodic "+ "chain of length " 
                      + str(L) + " with Jx,Jz = " + str((Jx,Jz)) + " is " + 
                      str(Chain(L, Jx, Jz, periodic).get_fundamental()))
    


        