# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:18:03 2018

@author: bmeyn
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

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
        
        self.p_mask = np.zeros((2*self.m_trotter, self.n_spins), dtype = int)
        for i in range(2 * self.m_trotter):
            for j in range(self.n_spins):
                if (i + j + 1)%2:
                    self.p_mask[i, j] = 1
        self.p_mask = self.p_mask.astype(bool)

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
        
        
        
    def copy(self,):
        copy = States(self.m_trotter, self.dtau, self.n_spins, self.Jx, self.Jz)
        copy.pattern = self.pattern.copy()
        return copy
        
    def spins_to_pattern(self):
        """
        Given the spin configuration, turn it into a pattern configuration. Allows the
        image to be created or the graph to be computed.
        """
        self.pattern = np.dot(self.p_left, np.dot(self.spins, self.p_right))

    def createimage(self):
        """
        Give the pattern representation of the configuration on the screen
        """
        
        #initializing the figure
        fig, ax = plt.subplots(figsize = (10,10))
        
        #this array corresponds to the image
        image = np.zeros((20*self.m_trotter*2,20*(self.n_spins)))
        
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
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
        pattern_energy = self.energymatrix[self.pattern]
        pattern_energy = pattern_energy[self.p_mask]
        energy = np.sum(pattern_energy)
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
        
        if self.spins[pos[0],pos[1] +1] != spin :
            return pos,False
        self.spins[pos[0],pos[1]] = (spin + 1)%2

        return [pos[0]+1,pos[1]],True
        
    
    def splitline(self):
        dE = 0
        dw = 1
        n  = rnd.randint(0,self.n_spins)
        print(n)
        #gd = int(rnd.rand()>0.5) # 0 means left spin from the case at stake, 1 right spin from the case at stake
        #print("randspin", n)
        p = [0,n] #[line, column, left or right]
        pattbefore = self.spins_to_pattern()
        print(self.spins.shape)
        for i in range(2*self.m_trotter):
            print("p",p)
            p,  has_changed = self.splitspin(p)
            print(p,has_changed)
            if has_changed == False :
                return dE, dw, False
        pattafter = self.spins_to_pattern()
        
        colg = n -1
        if n==0 :
            for i in range(2*self.m_trotter):
                dE += self.energymatrix[pattafter[i,n]] - self.energymatrix[pattafter[i,n]]
                dw *= self.weightmatrix[pattafter[i,n]] / (self.weightmatrix[pattafter[i,n]])
            
            
        elif n == self.n_spins-1:
            for i in range(2*self.m_trotter):
                dE += self.energymatrix[pattafter[i,n-1]] - self.energymatrix[pattafter[i,n-1]]
                dw *= self.weightmatrix[pattafter[i,n-1]] / (self.weightmatrix[pattafter[i,n-1]])
        
        else:
            for i in range(2*self.m_trotter):
                dE += self.energymatrix[pattafter[i,n-1]] - self.energymatrix[pattafter[i,n-1]] + self.energymatrix[pattafter[i,n]] - self.energymatrix[pattafter[i,n]]
                dw *= self.weightmatrix[pattafter[i,n-1]]*self.weightmatrix[pattafter[i,n]] / (self.weightmatrix[pattafter[i,n-1]] * self.weightmatrix[pattafter[i,n]])
            #print("p", p)
        #print("trysplit",n, dE, dw)
        return dE, dw, True
    
    
    def local_update_pos(self, pos):     
        
        """
        This method allows local updates, described in Fig.2 of the article. We will look for various type of 
        pattern, which are localised on four "white squares" in the pattern. We will call them the conf_down, 
        conf_up, conf_left, conf_right. 
        """
        
        
        
        
    
    
    def local_update(self,):
        dE = 0
        dw = 1
        
        spinpos  = rnd.randint(0,self.n_spins)
        mposprim  = 2*rnd.randint(0,self.m_trotter) + spinpos % 2
        mpos = rnd.randint(0,2*self.m_trotter)
        pos = np.array([mpos,spinpos])
        
        gd = (mpos + spinpos) % 2
        
        spin = self.spin[pos[0],pos[1]]
        spinup = self.spin[pos[0]+1,pos[1]]
        if spin == 0:
            return 0,1,False
        if spinup == 0:
            return 0,1,False
        
        if spinpos == 0 and mpos%2 ==0:
            return 0,1,False
        
        elif spinpos == n_spins-1:
            if n_spins%2== mpos%2:
                return 0,1,False
            
                
    
        else :
            if (mpos + spinpos) % 2 ==0 :
                if self.spin[pos[0],pos[1]-1] == 1 or self.spin[pos[0]+1,pos[1]-1] == 1 :
                    return 0,1,False
                self.spin[pos[0],pos[1]-1] =1 
                self.spin[pos[0]+1,pos[1]-1] = 1
                self.spin[pos[0],pos[1]] =0
                self.spin[pos[0]+1,pos[1]] = 0
            
            
            
            
            
            if (mpos + spinpos) % 2 ==1 :
                if self.spin[pos[0],pos[1]+1] == 1 or self.spin[pos[0]+1,pos[1]+1] == 1 :
                    return 0,1,False
                self.spin[pos[0],pos[1]+1] =1 
                self.spin[pos[0]+1,pos[1]+1] = 1
                self.spin[pos[0],pos[1]] =0
                self.spin[pos[0]+1,pos[1]] = 0
            
            
        
        
        
        
        
        
        dE,dw,has_changed = self.local_update_pos(pos)
        #print("try",pos,dE, dw)
#        while (has_changed == False and i-i_init<self.n_spins*2*self.m_trotter ):
#            i+=2
##            spinpos  = rnd.randint(0,self.n_spins)
##            mpos  = 2*rnd.randint(0,self.m_trotter) + spinpos % 2
#            spinpos  = i%self.n_spins
#            mpos  = (i//self.n_spins)%(2*self.m_trotter)
#            if(spinpos + mpos % 2 ==1):
#                spinpos += 1
#                spinpos = spinpos%self.n_spins
#                i+=1
#            pos = np.array([mpos,spinpos])
#            dE,dw,has_changed =self.local_update_pos(pos)
            #print("trylocal",pos,dE, dw, has_changed)

        return dE,dw,has_changed
    
    def basic_move_simple(self,n_splitline,n_localupdate): # always accept the change
        dw = 1
        dE = 0
        for i in range(n_splitline):
            dEtrans, dwtrans = self.splitline()
            #print("line split")
            dE += dEtrans
            dw *= dwtrans
        for j in range(n_localupdate):
            dEtrans, dwt = self.local_update()
            #print("locally updated")
            dE += dEtrans
            dw *= dwtrans
            
    def basic_move(self,n_splitline,n_localupdate):
        dw = 1
        dE = 0
        test = self.copy()
        for i in range(n_splitline):
            dEtrans, dwtrans,_ = test.splitline()
            #print("line split")
            dE += dEtrans
            dw *= dwtrans
            #print("split",dE, dw)
        for j in range(n_localupdate):
            dEtrans, dwtrans = test.local_update()
            #print("locally updated")
            dE += dEtrans
            dw *= dwtrans
            #print("loc",dE, dw)
        #print("fin",dE, dw)
        choice = rnd.random()
        if (dw>choice):
            self.pattern = test.pattern
            #print("change accepted",dE, dw)
            return dE, dw
        return 0 ,1
    
    def stoch_move(self,threshold):
        self.n_change +=1
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
            return 0,1
        if (dw>b):
            self.pattern = test.pattern
            #print("change accepted "+mtype,dE, dw)
            self.n_accepted += 1
            if mtype == "splitline":
                self.n_accsplitline += 1
            if mtype == "local":
                self.n_acclocal +=1
            return dE, dw
        #print("aborted",mtype)
        return 0 ,1
        

#    def local_update(self):
#        #introducing randomness
#        i = rnd.randint(0, self.m_trotter*self.n_spins)
#        i *= 2
#        #getting random position on the white squares
#        x = i // self.n_spins 
#        y = i % self.n_spins + x%2
#
#        dE, dw = self.local_update_pos(np.array([x,y], dtype = int))


#        return dE, dw
        
    
        
        
        
    
#    def Quantum_Monte_Carlo(self,n_warmup=100,n_cycles = 10000,length_cycle = 100):
#        energ = np.zeros(n_cycles)
#        # Monte Carlo simulation
#        for n in range(n_warmup+n_cycles):
#            # Monte Carlo moves
#            for l in range(length_cycle):
#                self.splitline
#                #self.autremodif
#            # measures
#            if n >= n_warmup:
#                energ[n-n_warmup] = self.total_energy()
#        return energ


#class config:
#    """A Feynman path in position/imaginary time space"""
#
#    def __init__(self, m_trotter, dtau, n_spins, Jx, Jz, mode = 'local', ):
#        self.m_trotter = m_trotter #division of the temporary time"
#        self.dtau = dtau
#        self.n_spins = n_spins
#        self.Jx = Jx
#        self.Jz = Jz
#        self.mode = mode
#
#
#    def compute_energy_autocorrelation(self,n_splitline,n_localupdate):
#        if(self.mode is 'local'):
#            s = States(self.m_trotter,self.dtau,self.n_spins,self.Jx,self.Jz)
#            for i in range (100): # warm
#                s.basic_move_simple(n_splitline,n_localupdate)
#            E = s.total_energy()
#            En_list = [E]
#            for k in range (15): # compute energie evolution
#                dE, _ = s.basic_move(n_splitline, n_localupdate)
#                E +=dE
#                En_list += [E]
#            Energies = np.array(En_list[:10])
#            sqmean = np.mean(Energies * Energies)
#            meansq = np.mean(Energies)**2
#            autocorr = np.zeros(5)
#            for t in range (5):
#                Energies_tmove = (En_list[t:t+10])
#                corr = np.mean(Energies * Energies_tmove)
#                autocorr[t] = (corr - meansq)/(sqmean - meansq)
#            x=np.linspace(0,4,5)
#            plt.plot(x,autocorr)
#            print("enelist",En_list)
#            print("Energies",Energies)
#            print("sqmean",sqmean)
#            print("meansq",meansq)
#            return
#            
#            
#        if(self.mode is 'loop'):
#            return

#    def basic_move(self,n_splitline,n_localupdate):
#        dw = 0
#        dE = 0
#        for i in range(n_splitline):
#            dEtrans, dwtrans = self.splitline()
#            dE += dEtrans
#            dw *= dwtrans
#        for j in range(n_localupdate):
#            dEtrans, dwt = self.local_update()
#            dE += dEtrans
#            dw *= dwtrans
#        return dE, dw
            
