# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:18:03 2018

@author: bmeyn
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
rnd.seed(10)

class States:
    """A spin tab in position/imaginary time space"""

    def __init__(self, m_trotter, dtau, n_spins, Jx, Jz, mode = "fixed"):
        """Construct a path at inverse temperature beta in dim spatial
           dimensions and ntau imaginary time points"""

       #division of the imaginary time
        self.m_trotter = m_trotter 
        self.dtau = dtau
        
        #number of spins along the chain
        self.n_spins = n_spins
        
        #interactions
        self.Jx = Jx
        self.Jz = Jz
        
        #this matrix describes the state of the spins along the 2D pattern
        #if self.spins[i, j] describes the spin at space = i and time = j
        self.spins = np.zeros((2*m_trotter, n_spins), dtype = int)
        
        #this matrix allows the computation of the energy and the weight 
        #it also allows the representation of the pattern given in the
        #Article (with a grey and white board, each black line is a line
        #of up spins at the corner of the tiles)
        self.pattern = np.zeros((2*m_trotter,n_spins-1), dtype = int)
        
        self.p_right = np.diag(np.ones(self.n_spins, dtype = int)) + np.diag(2 * np.ones(self.n_spins - 1, dtype = int), k = -1) + np.diag([2], k = self.n_spins - 1)
        self.p_left = np.diag(3 * np.ones(2 * self.m_trotter, dtype = int)) + np.diag(np.ones(2 * self.m_trotter - 1, dtype = int), k = 1) + np.diag([1], k = -2 * self.m_trotter + 1)
        
        self.p_mask = np.zeros((2*self.m_trotter, self.n_spins-1), dtype = int)
        self.black_mask = np.ones((2*self.m_trotter, self.n_spins-1), dtype = int)
        for i in range(2 * self.m_trotter):
            for j in range(self.n_spins-1):
                if (i + j + 1)%2:
                    self.p_mask[i, j] = 1
                    self.black_mask[i,j] = 0
        self.p_mask = self.p_mask.astype(bool) 
        self.black_mask = self.black_mask.astype(bool) #select the balck case

        #computing the energy depending on the configuration of the tiles.
        #Each tile has a particular energy, and one has to sum over the white tiles 
        #to find the energy. This is why we adopted the "pattern" representation
        self.a = self.Jz/4
        self.th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        self.coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        self.energymatrix = (-1/self.m_trotter)*np.array([-self.a, 0, 0, 0,
                                      self.a+self.th, self.a+self.coth, 0,
                                      self.a+self.coth, self.a+self.th, 
                                      0, 0, 0, -self.a])
    
        #computing the weight depending on the configuration of the tiles. 
        #Each tile has a particular weight, and one has to make the product over the white tiles 
        #to find the weight.
        self.b = np.exp(self.dtau*self.Jz/4)
        self.cosh = np.cosh(self.dtau*self.Jx/2)
        self.sinh = np.sinh(self.dtau*self.Jx/2)
        self.weightmatrix = np.array([1/self.b, 1., 1., 1.,
                                      self.b*self.cosh, self.b*self.sinh, 1.,
                                      self.b*self.sinh, self.b*self.cosh, 
                                      1., 1., 1., 1/self.b])
        
        #initializing the image
        self.greycase = np.ones((20,20),dtype=np.uint8) * 130
        self.case1 = np.ones((20,20))*255
        self.case2 = np.ones((20,20))*255
        self.case3 = np.ones((20,20),dtype=np.uint8)*255
        self.case4 = np.ones((20,20))*255
        self.case5 = np.ones((20,20))*255
        self.case6 = np.ones((20,20))*255
        self.case2[:,:2]=0
        self.case2[:,18:]=0
        for i in range(19):
            self.case3[i,19-i]=0
            self.case3[i,18-i]=0
            self.case4[i,i]=0
            self.case4[i,i+1]=0
        self.case6[:,:2]=0
        self.case5[:,18:]=0
        self.cases = np.array([self.case1, self.greycase, self.greycase, 
                          self.greycase, self.case6, self.case3,
                          self.greycase, self.case4, self.case5,
                          self.greycase, self.greycase, self.greycase, self.case2])
        
        
        
    def copy(self,): #make a copy of the state
        copy = States(self.m_trotter, self.dtau, self.n_spins, self.Jx, self.Jz)
        copy.pattern = self.pattern.copy()
        copy.spins = self.spins.copy()
        return copy
    def spinTostring(self):
        s = ''
        for i in range (2*self.m_trotter):
            for j in range (self.n_spins):
                s+= str(self.spins[i,j])
        return s
        
    def spins_to_pattern(self):
        """
        Given the spin configuration, turn it into a pattern configuration. Allows the
        image to be created or the graph to be computed.
        """

        self.pattern = np.dot(self.p_left, np.dot(self.spins, self.p_right))
        self.pattern = self.pattern[:,:self.n_spins-1]

        self.pattern[self.black_mask]=1 #give value one to the black case 1 is associated to an energy of 0 and a weight of 1

    def createimage(self):
        """
        Give the pattern representation of the configuration on the screen
        """
        
        #initializing the figure
        fig, ax = plt.subplots(figsize = (10,10))
        
        #this array corresponds to the image
        image = np.zeros((20*self.m_trotter*2,20*(self.n_spins-1)))
        
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins-1):
                if((i+j+1)%2):
                    tile = self.pattern[i, j]
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=self.cases[tile]
                else:
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=130
        
        image = np.array(image,dtype=np.uint8)
        ax.imshow(image, cmap = "Greys_r")
            
        
    def total_energy(self):
        """
        Computes the Energy of the configuration. Uses self.pattern to know the tiles
        Then uses self.energymatrix to know the energy of each tile. Sum over them.
        """
        #open_pattern = self.pattern[:,:self.n_spins-2]
        pattern_energy = self.energymatrix[self.pattern]
        pattern_energy = pattern_energy[self.p_mask] #take only the energy of white case
        energy = np.sum(pattern_energy)
        
#        pattern_energy = self.energymatrix[open_pattern]
#        pattern_energy = pattern_energy[self.p_mask]
#        energy = np.sum(pattern_energy)
        return energy
    
    
    def weight(self):
        """
        Computes the Weight of the configuration. Uses self.pattern to know the tiles
        Then uses self.weightmatrix to know the weight of each tile. Make the product of them.
        """
        pattern_weight = self.weightmatrix[self.pattern]
        pattern_weight = pattern_weight[self.p_mask]
        weight = np.prod(pattern_weight)
        return weight
    
    
    def splitspin(self,pos):
        
        energymatrix = self.energymatrix
        weightmatrix = self.weightmatrix
        spin = self.spins[pos[0],pos[1]]
            
        if (self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]] != spin and pos[0]!=2*self.m_trotter-1) :
            return pos,False
        self.spins[pos[0],pos[1]] = (spin + 1)%2

        return [pos[0]+1,pos[1]],True
        
    
    def splitline(self): #change a line 
        dE = 0
        dw = 1
        n  = rnd.randint(0,self.n_spins) #choose line
        

        p = [0,n] #[line, column, left or right]
        self.spins_to_pattern()
        pattbefore = self.pattern #save pattern
        


        for i in range(2*self.m_trotter):

            p,  has_changed = self.splitspin(p) #change spins one by one

            if has_changed == False :
                return dE, dw, False
        self.spins_to_pattern()

        pattafter = self.pattern #look at the pattern after
        
        
        

#        lets compute dE and dw now

        if n==0 : #left border
            for i in range(2*self.m_trotter):
                dE += self.energymatrix[pattafter[i,n]] - self.energymatrix[pattbefore[i,n]]
                dw *= self.weightmatrix[pattafter[i,n]] / (self.weightmatrix[pattbefore[i,n]])

            
            
        elif n == self.n_spins-1: #right border
            for i in range(2*self.m_trotter):

                dE += self.energymatrix[pattafter[i,n-1]] - self.energymatrix[pattbefore[i,n-1]]
                dw *= self.weightmatrix[pattafter[i,n-1]] / (self.weightmatrix[pattbefore[i,n-1]])
        
        else: #middle of the board
            for i in range(2*self.m_trotter):
                
                dE += self.energymatrix[pattafter[i,n-1]] - self.energymatrix[pattbefore[i,n-1]] + self.energymatrix[pattafter[i,n]] - self.energymatrix[pattbefore[i,n]]
                dw *= self.weightmatrix[pattafter[i,n-1]]*self.weightmatrix[pattafter[i,n]] / (self.weightmatrix[pattbefore[i,n-1]] * self.weightmatrix[pattbefore[i,n]])

        return dE, dw, True
    
      
        
    
    def local_update(self):
        """
        This method allows local updates, described in Fig.2 of the article. We will look for various type of 
        pattern, which are localised on four "white squares" in the pattern around one black. 
        """
        
        
        
        
        dE = 0
        dw = 1
        
        spinpos  = rnd.randint(0,self.n_spins)

        mpos = rnd.randint(0,2*self.m_trotter)
        
        pos = np.array([mpos,spinpos])

        
        spin = self.spins[pos[0],pos[1]]
        spinup = self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]]
        if spin == 0:
            #print("spindown")
            return 0,1,False
        if spinup == 0:

            return 0,1,False
        
        
         # check if it is not going to change spin out of bound at left (spin < 0)       
        if spinpos == 0 and mpos%2 ==0:

            return 0,1,False
        
        
        # check if it is not going to change spin out of bound at right (spin > n_spin)
        if spinpos == self.n_spins-1: 
            
            if self.n_spins%2 == mpos%2:

                return 0,1,False
            
                
    
        
        #print("enter")
        if (mpos + spinpos) % 2 ==0 :  #am I at the right or the left of the black case

            if self.spins[pos[0],pos[1]-1] == 1 or self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]-1] == 1 :

                return 0,1,False
            
            patt_mpos= mpos
            patt_spinpos = spinpos - 1
            self.spins_to_pattern()
             # je calcule les energie et poids des neufs cases autour de la case noir qui m'interesse, si je suis sur un bord je prend celle de l'autre cote (comme si on etait en periodique)
             #mais comme je fais une difference avec apres modif ces termes disparaissent
            Ebefore = self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
            wbefore = self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
            
            
        
            
            self.spins[pos[0],pos[1]-1] =1 
            self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]-1] = 1
            self.spins[pos[0],pos[1]] =0
            self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]] = 0
            

            
            self.spins_to_pattern()
            Eafter = self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
            wafter = self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
        
        
        
        
        
        if (mpos + spinpos) % 2 ==1 : #am I at the right or the left of the black case
            
            # position of the black case in the pattern description

            patt_mpos= mpos
            patt_spinpos = spinpos
            self.spins_to_pattern()
            Ebefore = self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
            wbefore = self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
            
            
            
            
            
            if self.spins[pos[0],pos[1]+1] == 1 or self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]+1] == 1 :

                return 0,1,False
            

            self.spins[pos[0],pos[1]+1] =1

            self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]+1] = 1
            self.spins[pos[0],pos[1]] =0
            self.spins[(pos[0]+1)%(2*self.m_trotter),pos[1]] = 0
            

            
            self.spins_to_pattern()

            
            Eafter = self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] + \
                    self.energymatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
            wafter = self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos-1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos-1)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos)%(self.n_spins-1)]] * \
                    self.weightmatrix[self.pattern[(patt_mpos+1)%(2*self.m_trotter),(patt_spinpos+1)%(self.n_spins-1)]]
                
        

        dE= Eafter - Ebefore
        dw = wafter / wbefore
        
        return dE, dw, True
            
            
        
        
        
        

    

#    
    def stoch_move(self,threshold):
        
        dw = 1
        dE = 0
        test = self.copy()
        a = rnd.rand()
        b = rnd.rand()
        #print(a)
        if (a<threshold):
            dEt,dwt, has_changed= test.local_update()
            dE += dEt
            dw *= dwt
            mtype = "local"
            #print(mtype, dw)
        else:
            dEt,dwt, has_changed= test.splitline()
            dE += dEt
            dw *= dwt
            mtype = "splitline"
            #print("try split b = ",b,"dw = ",dw)
#        dw2 = test.weight()/self.weight()
#        if (dw != dw2):
        #print("try", mtype, has_changed,dw)
        if has_changed == False:
            return 0,1
        if (dw>b):
            self.pattern = test.pattern
            self.spins = test.spins
            #print("change accepted "+mtype,dE, dw)


            return dE, dw
        #print("aborted",mtype)
        return 0 ,1
        

    def stoch_move_forced(self,threshold):
        
        dw = 1
        dE = 0
        test = self.copy()
        a = rnd.rand()
        b = rnd.rand()
        #print(a)
        if (a<threshold):
            dEt,dwt, has_changed= test.local_update()
            dE += dEt
            dw *= dwt
            mtype = "local"
            #print(mtype, dw)
        else:
            dEt,dwt, has_changed= test.splitline()
            dE += dEt
            dw *= dwt
            mtype = "splitline"
            #print("try split b = ",b,"dw = ",dw)
#        dw2 = test.weight()/self.weight()
#        if (dw != dw2):
#            print("alert", mtype, dw,dw2)
        
        if has_changed == False:
            #print(mtype, dw, has_changed)
            #self.createimage()
            return 0,1
        if (has_changed == True):
            self.pattern = test.pattern
            self.spins = test.spins
            #print(mtype, dw, has_changed)
            #self.createimage()
        
        return dE ,dw



