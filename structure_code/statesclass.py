# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:18:03 2018

@author: bmeyn
"""

import numpy as np
import numpy.random as rnd

class States:
    """A Feynman path in position/imaginary time space"""

    def __init__(self, m_trotter, dtau, n_spins, Jx, Jz):
        """Construct a path at inverse temperature beta in dim spatial
           dimensions and ntau imaginary time points"""

        self.m_trotter = m_trotter
        self.dtau = dtau
        self.n_spins = n_spins
        self.Jx = Jx
        self.Jz = Jz
        #self.spins_up = rnd.randint(0, n_spins+1)
        self.pattern = np.zeros((2*m_trotter,n_spins,6))
        for i in range (2*m_trotter):
            for j in range(n_spins-1):
                self.pattern[i,j,0] = 1
        
        self.spins_up = 0
    
    
#    def to_boxconfig(self):
#        new_writing = np.zeros((6,2*self.m_trotter))
#        for line in range(2*self.m_trotter):
#            for column in range(self.n_spins-1):
#                if (self.pattern[line,column]!=0):
#                    new_writing[self.pattern[line,column],line]+=1
#                    
#        return new_writing
                    
            
        
    def total_energy(self):
        
        a = self.Jz/4
        th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        coth = self.Jx/2*np.cotanh(self.dtau*self.Jx/2)
        energymatrix = np.array([-a, -a, a+coth, a+coth, a+th, a+th])
        energy = np.sum(self.pattern*energymatrix)
        return energy
    
    def weight(self):
        a = np.exp(self.dtau*self.Jz/4)
        cosh = np.cosh(self.dtau*self.Jx/2)
        sinh = np.sinh(self.dtau*self.Jx/2)
        weightmatrix = np.array([1/a, 1/a, -a*sinh, -a*sinh, a*cosh, a*cosh])
        weight = np.prod(self.pattern*weightmatrix)
        return weight
    
    def 
        