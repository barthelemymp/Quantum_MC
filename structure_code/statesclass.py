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
        self.pattern = np.zeros((2*m_trotter,n_spins))
        for i in range (2*m_trotter):
            for j in range(n_spins-1):
                self.pattern[i,j] = i+j % 2
        
        self.spins_up = 0
    
    
    def to_boxconfig(self):
        new_writing = np.zeros((6,2*self.m_trotter))
        for line in range(2*self.m_trotter):
            for column in range(self.n_spins-1):
                if (self.pattern[line,column]!=0):
                    new_writing[self.pattern[line,column],line]+=1
                    
        return new_writing
                    
            
        
    def total_energy(self):
        
        new_writing = self.to_boxconfig()
        a = self.dtau*self.Jz/4
        logco = np.log(np.cosh(self.dtau*self.Jx/2))
        logsi = np.log(np.sinh(self.dtau*self.Jx/2))
        energymatrix = np.diag([-a, -a, a*logsi, a*logsi, a*logco, a*logco])
        energy = np.sum(energymatrix.dot(new_writing))
        return energy
    
    def weight(self):
        new_writing = self.to_boxconfig()
        a = np.exp(self.dtau*self.Jz/4)
        cosh = np.cosh(self.dtau*self.Jx/2)
        sinh = np.sinh(self.dtau*self.Jx/2)
        weightmatrix = np.diag([1/a, 1/a, -a*sinh, -a*sinh, a*cosh, a*cosh])
        weight = np.sum(energymatrix.dot(new_writing))
        