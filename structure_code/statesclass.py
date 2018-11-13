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

        self.m_trotter = m_trotter #division of the temporary time"
        self.dtau = dtau
        self.n_spins = n_spins
        self.Jx = Jx
        self.Jz = Jz
        #self.spins_up = rnd.randint(0, n_spins+1)
        self.pattern = np.zeros((2*m_trotter,n_spins,6))
        self.pattern[:,:,:]=np.nan
        for i in range (2*m_trotter):
            for j in range(n_spins):
                self.pattern[i,j,0] = (i+j+1)%2
                if (self.pattern[i,j,0] == 0):
                    self.pattern[i,j,0] = np.nan
        
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
        coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        energymatrix = np.array([-a, -a, a+coth, a+coth, a+th, a+th])
        energy = np.nansum(self.pattern*energymatrix)
        print("en",energy)
        return energy
    
    def weight(self):
        a = np.exp(self.dtau*self.Jz/4)
        cosh = np.cosh(self.dtau*self.Jx/2)
        sinh = np.sinh(self.dtau*self.Jx/2)
        weightmatrix = np.array([1/a, 1/a, -a*sinh, -a*sinh, a*cosh, a*cosh])
        weight = np.nanprod(self.pattern*weightmatrix)
        return weight
    
    def splitspin(self,pos,dE): #probleme d'origine du dernier mouvement a regler 
        ae = self.Jz/4
        th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        energymatrix = np.array([-ae, -ae, ae+coth, ae+coth, ae+th, ae+th])
        
        print("pos",pos)
        a = np.zeros(6)
        a[:]=np.nan
        a[0]=1
        b = np.zeros(6)
        b[:]=np.nan
        b[1]=1
        c = np.zeros(6)
        c[:]=np.nan
        c[2]=1
        d = np.zeros(6)
        d[:]=np.nan
        d[3]=1
        e = np.zeros(6)
        e[:]=np.nan
        e[4]=1
        f = np.zeros(6)
        f[:]=np.nan
        f[5]=1
        conf = np.nanargmax(np.array(self.pattern[pos[0],pos[1],:])) + 1
        print("conf",conf)
        if(pos[2]==0):
            if(conf==1):
                self.pattern[pos[0],pos[1],:] = f
                dE += energymatrix[5]-energymatrix[0]
                return np.array([pos[0]+1,pos[1]-1,1])
            
            if(conf==2):
                self.pattern[pos[0],pos[1],:,] = e
                dE +=energymatrix[4]-energymatrix[1]
                return np.array([pos[0]+1,pos[1]-1,1])
            if(conf==3):
                self.pattern[pos[0],pos[1],:] = a
                dE += energymatrix[0]-energymatrix[2]
                return np.array([pos[0]+1,pos[1]+1,0])
            if(conf==4):
                self.pattern[pos[0],pos[1],:] = b
                dE += energymatrix[1]-energymatrix[3]
                return np.array([pos[0]+1,pos[1]+1,0])
            if(conf==5):
                self.pattern[pos[0],pos[1],:] = b
                dE +=energymatrix[1]-energymatrix[4]
                return np.array([pos[0]+1,pos[1]-1,1])
            if(conf==6):
                self.pattern[pos[0],pos[1],:] = a
                dE += energymatrix[0]-energymatrix[5]
                return np.array([pos[0]+1,pos[1]-1,1])     
        elif(pos[2]==1) :
            if(conf==1):
                self.pattern[pos[0],pos[1],:] = e
                dE +=energymatrix[4]-energymatrix[0]
                return np.array([pos[0]+1,pos[1]+1,0])
            if(conf==2):
                self.pattern[pos[0],pos[1],:] = f
                dE +=energymatrix[5]-energymatrix[1]
                return np.array([pos[0]+1,pos[1]+1,0])
            if(conf==3):
                self.pattern[pos[0],pos[1],:] = b
                dE +=energymatrix[1]-energymatrix[2]
                return np.array([pos[0]+1,pos[1]-1,1])
            if(conf==4):
                self.pattern[pos[0],pos[1],:] = a
                dE +=energymatrix[0]-energymatrix[3]
                return np.array([pos[0]+1,pos[1]-1,1])
            if(conf==5):
                self.pattern[pos[0],pos[1],:] = a
                dE +=energymatrix[0]-energymatrix[4]
                return np.array([pos[0]+1,pos[1]+1,0])
            if(conf==6):
                self.pattern[pos[0],pos[1],:] = b
                dE +=energymatrix[1]-energymatrix[5]
                return np.array([pos[0]+1,pos[1]+1,0]) 
        
        
    
    def splitline(self):
        dE = 0
        n  = int(rnd.randint(0,self.n_spins)/2)*2#attention derniere ligne a checker
        print("randspin", n)
        p = [0,n,0]
        for i in range(2*self.m_trotter):
            p = self.splitspin(p,dE)
            
            print("p", p)
        return dE
    
    def local_update_pos(self, pos):
        ae = self.Jz/4
        th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        energymatrix = np.array([-ae, -ae, ae+coth, ae+coth, ae+th, ae+th])
        
        a = np.zeros(6)
        a[:]=np.nan
        a[0]=1
        b = np.zeros(6)
        b[:]=np.nan
        b[1]=1
        c = np.zeros(6)
        b[:]=np.nan
        c[2]=1
        d = np.zeros(6)
        d[:]=np.nan
        d[3]=1
        e = np.zeros(6)
        e[:]=np.nan
        e[4]=1
        f = np.zeros(6)
        f[:]=np.nan
        f[5]=1
        conf = np.nanargmax(np.array(self.pattern[pos[0],pos[1],:])) + 1
        
        
        if (pos[2]) == 0:
            
        
        
        
        
        
        
    
    def Quantum_Monte_Carlo(self,n_warmup=100,n_cycles = 10000,length_cycle = 100):
        energ = np.zeros(n_cycles)
        # Monte Carlo simulation
        for n in range(n_warmup+n_cycles):
            # Monte Carlo moves
            for l in range(length_cycle):
                self.splitline
                #self.autremodif
            # measures
            if n >= n_warmup:
                energ[n-n_warmup] = self.total_energy()
        return energ
    
                
                    