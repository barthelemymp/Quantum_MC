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
    
    
    def __init__(self, L, Jx, Jz, beta, periodic = True):
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
        #beta
        self.beta = beta
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
        self.states = 1/np.sqrt(self.L) * self.states[:, 8-self.L:8]
        
        #energies of eigen states
        self.energies = np.array([])
        
        self.softmax = np.array([])
    
      
    #For a chain, the hamiltonian can be separated in halmitonians
    #only acting on pairs.
    
    def compute_hamiltonian_on_pair(self, k, state_i, state_j):
        """
        This method compute hamiltonian_two_sites | state >
        """
        
        pair = np.array([state_i[k], state_i[ (k+1) % self.L ]])
                
        #instanciate
        upup = 1/np.sqrt(self.L) * np.array([1.,1.])
        downdown = - upup
        updown = 1/np.sqrt(self.L) * np.array([1.,-1.])
        downup = - updown
        
        state_flip = np.array([])
        
        #if up - up, the element matrix is non zero only for pair_state_j = up - up
        if np.array_equal(pair, upup):
            return self.Jz/4 * np.array_equal(state_i, state_j)
        #if down - down
        elif np.array_equal(pair, downdown):
#            print("dd")
            return self.Jz/4 * np.array_equal(state_i, state_j)
        #if up - down, either pair_state_j is down - up or up -down
        elif np.array_equal(pair, updown):
#            print("updown")
            
            if k < self.L - 1:
                state_flip = np.concatenate((state_i[:k], downup, state_i[k+2:]))
            elif (k == self.L - 1) and self.periodic:
                state_flip = np.concatenate((np.array([downup[1]]), state_i[1:k], np.array([downup[0]])))
            return ( -self.Jz/4 * np.array_equal(state_i, state_j) +
                     self.Jx/2 * np.array_equal(state_flip, state_j) )
        #if down - up
#        print("downup")
        if k < self.L - 1:
            state_flip = np.concatenate((state_i[:k], updown, state_i[k+2:]))
        elif (k == self.L - 1) and self.periodic:
            state_flip = np.concatenate((np.array([updown[1]]), state_i[1:k], np.array([updown[0]])))
#        print(state_flip, state_j)
#        print(np.array_equal(state_flip, state_j))
        return  ( -self.Jz/4 * np.array_equal(state_i, state_j) +
                  self.Jx/2 * np.array_equal(state_flip, state_j) )
    

    def compute_element_matrix(self, i, j):
        """
        This method compute the matrix even element between state i and j using
        self.compute_hamiltonian_on_pair()
        """
        #let get i and j
        state_i = self.states[i]
        state_j = self.states[j]
#        print(state_i, state_j)
        
        #result
        sum = 0.
        
        #let compute the pair to pair hamiltonian
        for k in range(0, self.L):
            sum += self.compute_hamiltonian_on_pair(k, state_i, state_j)
#            print(k, sum)

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
        self.energies = np.real(np.linalg.eigvalsh(self.hamiltonian))
        return self.energies
    
    def get_fundamental(self):
        self.set_eigenvalues()
        return np.min(self.energies)
    
    def set_softmax(self):
        ex = np.exp( - self.beta * self.energies )
        sum_ex = np.sum( ex )
        self.softmax = ex / sum_ex
    
    def get_mean_energy(self):
        self.set_eigenvalues()
        self.set_softmax()
        return np.dot(self.energies, self.softmax)
    

c = Chain(4, 1, 1, 4, periodic = True)
c.set_eigenvalues()
print(c.energies)

def compute_fundamental_chain(L, Jx, Jz, s = 'result_exact_computation.txt', periodic = True):
    with open(s, 'w') as fichier:
        fichier.write("The fundamental's energy for a " + periodic*"periodic "+ "chain of length " 
                      + str(L) + " with Jx,Jz = " + str((Jx,Jz)) + " is " + 
                      str(Chain(L, Jx, Jz, periodic).get_fundamental()))

def compute_energies(L, Jx, Jz, periodic = True):
    with open('energies' + str((L, Jx, Jz)) + '.txt', 'w') as fichier:
        fichier.write("The energies for a " + periodic*"periodic "+ "chain of length " 
                      + str(L) + " with Jx,Jz = " + str((Jx,Jz)) + " is " + 
                      str(Chain(L, Jx, Jz, 0, periodic).set_eigenvalues()))

def compute_mean_chain_Jx(L, Jz, beta, periodic = True ):
    with open('MeanEnergyJx' + str((Jz, beta)) + '.txt', 'w') as fichier:
        for Jx in range(0, 5):
            fichier.write("The mean energy for a " + periodic*"periodic "+ "chain of length " 
                          + str(L) + " with Jx,Jz, beta = " + str((-Jx,Jz, beta)) + " is " + 
                          str(Chain(L, -Jx, Jz, beta, periodic).get_mean_energy()) + '\n')
    

    
def moyenneenergy2(beta):
    Z = 2*np.exp(beta*0.5) + 2*np.exp(-beta*0.5)
    Emo = -np.exp(beta*0.5) + np.exp(-beta*0.5)
    return Emo/Z

def moyenneenergy4(beta):
    Z = 2*np.exp(beta*1) + 2*np.exp(-beta*1)+12
    Emo = -2*np.exp(beta*1) + 2*np.exp(-beta*1)
    return Emo/Z

        