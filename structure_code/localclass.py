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

        self.a = self.Jz/4
        self.th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        self.coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        self.energymatrix = (-1/m_trotter)*np.array([-self.a, -self.a, self.a+self.coth, self.a+self.coth, self.a+self.th, self.a+self.th])

        self.b = np.exp(self.dtau*self.Jz/4)
        self.cosh = np.cosh(self.dtau*self.Jx/2)
        self.sinh = np.sinh(self.dtau*self.Jx/2)
        self.weightmatrix = np.array([1/self.b, 1/self.b, -self.b*self.sinh, -self.b*self.sinh, self.b*self.cosh, self.b*self.cosh])
        
        
    def copy(self,):
        copy = States(self.m_trotter, self.dtau, self.n_spins, self.Jx, self.Jz)
        copy.pattern = self.pattern.copy()
        
        copy.spins_up = self.spins_up
        return copy
        

    def createimage(self, casesize=20):
        fig, ax = plt.subplots(figsize = (10,10))
        greycase = np.ones((20,20),dtype=np.uint8) * 70
        case1 = np.ones((20,20))*255
        case2 = np.ones((20,20))*255
        case3 = np.ones((20,20),dtype=np.uint8)*255
        case4 = np.ones((20,20))*255
        case5 = np.ones((20,20))*255
        case6 = np.ones((20,20))*255
        
        case2[:,:2]=0
        case2[:,18:]=0
        
        for i in range(19):
            case3[i,19-i]=0
            case3[i,18-i]=0
            case4[i,i]=0
            case4[i,i+1]=0
            
        case6[:,:2]=0
        
        case5[:,18:]=0
        cases = [case1,case2,case3,case4,case5,case6,greycase]
        
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
                if((i+j+1)%2):
                    conf = np.nanargmax(np.array(self.pattern[i,j,:]))
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=cases[conf]
                else:
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=130
                    
                
        image = np.array(image,dtype=np.uint8)
        ax.imshow(image, cmap = 'Greys_r')

        
    
#    def to_boxconfig(self):
#        new_writing = np.zeros((6,2*self.m_trotter))
#        for line in range(2*self.m_trotter):
#            for column in range(self.n_spins-1):
#                if (self.pattern[line,column]!=0):
#                    new_writing[self.pattern[line,column],line]+=1
#                    
#        return new_writing
                    
            
        
    def total_energy(self):
        energy = np.nansum(self.pattern*self.energymatrix)
        return energy
    
    def weight(self):
        weight = np.nanprod(self.pattern*self.weightmatrix)
        return weight
    
    def splitspin(self,pos,dE,dw):  
        #getting the weight matrix
        energymatrix = self.energymatrix
        weightmatrix = self.weightmatrix
        
        #print("pos",pos)
        conf1 = np.zeros(6)
        conf1[:]=np.nan
        conf1[0]=1
        conf2 = np.zeros(6)
        conf2[:]=np.nan
        conf2[1]=1
        conf3 = np.zeros(6)
        conf3[:]=np.nan
        conf3[2]=1
        conf4 = np.zeros(6)
        conf4[:]=np.nan
        conf4[3]=1
        conf5 = np.zeros(6)
        conf5[:]=np.nan
        conf5[4]=1
        conf6 = np.zeros(6)
        conf6[:]=np.nan
        conf6[5]=1
        conf = np.nanargmax(np.array(self.pattern[pos[0],pos[1],:])) + 1
        #print("conf",conf)
        if(pos[2]==0):
            if(conf==1):
                self.pattern[pos[0],pos[1],:] = conf6
                dE += energymatrix[5]-energymatrix[0]
                dw *= weightmatrix[5]/weightmatrix[0]
                return np.array([pos[0]+1,(pos[1]-1)%self.n_spins,1]), dE, dw
            
            if(conf==2):
                self.pattern[pos[0],pos[1],:,] = conf5
                dw *= weightmatrix[4]/weightmatrix[1]
                dE +=energymatrix[4]-energymatrix[1]
                return np.array([pos[0]+1,(pos[1]-1)%self.n_spins,1]), dE, dw
            if(conf==3):
                self.pattern[pos[0],pos[1],:] = conf1
                dw *= weightmatrix[0]/weightmatrix[2]
                dE += energymatrix[0]-energymatrix[2]
                return np.array([pos[0]+1,(pos[1]+1)%self.n_spins,0]), dE, dw
            if(conf==4):
                self.pattern[pos[0],pos[1],:] = conf2
                dw *= weightmatrix[1]/weightmatrix[3]
                dE += energymatrix[1]-energymatrix[3]
                return np.array([pos[0]+1,(pos[1]+1)%self.n_spins,0]), dE, dw
            if(conf==5):
                self.pattern[pos[0],pos[1],:] = conf2
                dw *= weightmatrix[1]/weightmatrix[4]
                dE +=energymatrix[1]-energymatrix[4]
                return np.array([pos[0]+1,(pos[1]-1)%self.n_spins,1]), dE, dw
            if(conf==6):
                self.pattern[pos[0],pos[1],:] = conf1
                dw *= weightmatrix[0]/weightmatrix[5]
                dE += energymatrix[0]-energymatrix[5]
                return np.array([pos[0]+1,(pos[1]-1)%self.n_spins,1]), dE, dw     
        elif(pos[2]==1) :
            if(conf==1):
                self.pattern[pos[0],pos[1],:] = conf5
                dw *= weightmatrix[4]/weightmatrix[0]
                dE +=energymatrix[4]-energymatrix[0]
                return np.array([pos[0]+1,(pos[1]+1)%self.n_spins,0]), dE, dw
            if(conf==2):
                self.pattern[pos[0],pos[1],:] = conf6
                dw *= weightmatrix[5]/weightmatrix[1]
                dE +=energymatrix[5]-energymatrix[1]
                return np.array([pos[0]+1,(pos[1]+1)%self.n_spins,0]), dE, dw
            if(conf==3):
                self.pattern[pos[0],pos[1],:] = conf2
                dw *= weightmatrix[1]/weightmatrix[2]
                dE +=energymatrix[1]-energymatrix[2]
                return np.array([pos[0]+1,(pos[1]-1)%self.n_spins,1]), dE, dw
            if(conf==4):
                self.pattern[pos[0],pos[1],:] = conf1
                dw *= weightmatrix[0]/weightmatrix[3]
                dE +=energymatrix[0]-energymatrix[3]
                return np.array([pos[0]+1,(pos[1]-1)%self.n_spins,1]), dE, dw
            if(conf==5):
                self.pattern[pos[0],pos[1],:] = conf1
                dw *= weightmatrix[0]/weightmatrix[4]
                dE +=energymatrix[0]-energymatrix[4]
                return np.array([pos[0]+1,(pos[1]+1)%self.n_spins,0]), dE, dw
            if(conf==6):
                self.pattern[pos[0],pos[1],:] = conf2
                dw *= weightmatrix[1]/weightmatrix[5]
                dE +=energymatrix[1]-energymatrix[5]
                return np.array([pos[0]+1,(pos[1]+1)%self.n_spins,0]), dE, dw 
        
        
    
    def splitline(self):
        dE = 0
        dw = 1
        n  = int(rnd.randint(0,self.n_spins)/2)*2#attention derniere ligne a checker
        gd = int(rnd.rand()>0.5) # 0 means left spin from the case at stake, 1 right spin from the case at stake
        #print("randspin", n)
        p = [0,n,gd] #[line, column, left or right]
        for i in range(2*self.m_trotter):
            p, dE, dw = self.splitspin(p,dE,dw)
            
            #print("p", p)
        #print("trysplit",n, dE, dw)
        return dE, dw
    
    
    def local_update_pos(self, pos):       
        """
        This method allows local updates, described in Fig.2 of the article. We will look for various type of 
        pattern, which are localised on four "white squares" in the pattern. We will call them the conf_down, 
        conf_up, conf_left, conf_right. 
        """
        energymatrix = self.energymatrix
        weightmatrix = self.weightmatrix

        dE = 0
        dw = 1
        
        #here are the various allowed conf in our pattern for each white square
        conf1 = np.zeros(6)
        conf1[:]=np.nan
        conf1[0]=1
        conf2 = np.zeros(6)
        conf2[:]=np.nan
        conf2[1]=1
        conf3 = np.zeros(6)
        conf3[:]=np.nan
        conf3[2]=1
        conf4 = np.zeros(6)
        conf4[:]=np.nan
        conf4[3]=1
        conf5 = np.zeros(6)
        conf5[:]=np.nan
        conf5[4]=1
        conf6 = np.zeros(6)
        conf6[:]=np.nan
        conf6[5]=1
        
        #print("pos",pos)
        
        #we get the conf of the white squares we are interested in
        conf_down = np.nanargmax(self.pattern[pos[0],pos[1],:]) + 1
        conf_up = np.nanargmax(self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                                              pos[1],\
                                              :]) + 1
        conf_left = np.nanargmax(self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                                ( pos[1] - 1 )%self.n_spins,\
                                                :]) + 1
        conf_right = np.nanargmax(self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                                ( pos[1] + 1 )%self.n_spins,\
                                                :]) + 1
    
        #we can eliminate the cases in which the first spins are up-up or down-down
        if conf_down < 3: 
            return (dE, dw, False)
        #the bottom square is up-down==>down-up   
        elif conf_down == 3:
            #the up square must be down-up==>up-down
            if conf_up != 4:
                return (dE, dw, False)
            #then we have two possibilities for the left square, either up-down==>up-down or down-down==>down-down
            elif (conf_left != 1 and conf_left!= 6):
                return (dE, dw, False)
            #then we have two possibilities for the right square, either up-down==>up-down or up-up==>up-up
            elif (conf_right != 2 and conf_right!= 6):
                return (dE, dw, False)
            #we move the spins from up-down==>down-up==>down-up==>up-down to 
            #up-down==>up-down==>up-down==>up-down as describe in the local update
            else:
                self.pattern[pos[0],pos[1],:] = conf6
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf6
                dE += 2*energymatrix[4] - 2*energymatrix[2]
                dw *= weightmatrix[4]**2 / weightmatrix[2]**2
                #moving the left white square
                if conf_left == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf5
                    dE += energymatrix[4] - energymatrix[0]
                    dw *= weightmatrix[4] / weightmatrix[0]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf2
                    dE += energymatrix[1] - energymatrix[5]
                    dw *= weightmatrix[1] / weightmatrix[5]
                #moving the right white square
                if conf_right == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf5
                    dE += energymatrix[4] - energymatrix[1]
                    dw *= weightmatrix[4] / weightmatrix[1]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf1
                    dE += energymatrix[0] - energymatrix[5]
                    dw *= weightmatrix[0] / weightmatrix[5]
                return (dE, dw,True)
        #case in which the bottom square is down-up==>up-down
        elif conf_down == 4:
            if conf_up != 3:
                return (dE, dw,False)
            elif (conf_left !=2 and conf_left!=5):
                return (dE, dw,False)
            elif (conf_right !=1 and conf_right!=5):
                return (dE, dw,False)
            else:
                self.pattern[pos[0],pos[1],:] = conf5
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf5
                dE += 2 * (energymatrix[4] - energymatrix[2])
                dw *= (weightmatrix[4] / weightmatrix[2]) ** 2
                #moving the left white square
                if conf_left == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf6
                    dE += energymatrix[5] - energymatrix[1]
                    dw *= weightmatrix[5] / weightmatrix[1]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf1
                    dE += energymatrix[0] - energymatrix[4]
                    dw *= weightmatrix[0] / weightmatrix[4]
                #moving the right white square
                if conf_right == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf6
                    dE += energymatrix[5] - energymatrix[0]
                    dw *= weightmatrix[5] / weightmatrix[0]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf2
                    dE += energymatrix[1] - energymatrix[4]
                    dw *= weightmatrix[1] / weightmatrix[4]
                
                return (dE, dw,True)
        #case in which the bottom square is down-up==>down-up
        elif conf_down == 5:
            if conf_up != 5:
                return (dE, dw, False)
            elif (conf_left != 1 and conf_left != 6):
                return (dE, dw, False)
            elif (conf_right != 2 and conf_right != 6):
                return (dE, dw, False)
            else:
                self.pattern[pos[0],pos[1],:] = conf4
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf3
                dE += 2 * (energymatrix[2] - energymatrix[4])
                dw *= (weightmatrix[2] / weightmatrix[4]) ** 2
                #moving the left white square
                if conf_left == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf5
                    dE += energymatrix[4] - energymatrix[0]
                    dw *= weightmatrix[4] / weightmatrix[0]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf2
                    dE += energymatrix[1] - energymatrix[5]
                    dw *= weightmatrix[1] / weightmatrix[5]
                #moving the right white square
                if conf_right == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf5
                    dE += energymatrix[4] - energymatrix[1]
                    dw *= weightmatrix[4] / weightmatrix[1]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf1
                    dE += energymatrix[0] - energymatrix[5]
                    dw *= weightmatrix[0] / weightmatrix[5]
                    
                return (dE, dw, True)
        #case in which the bottom square is updown==>up-down
        else:
            if conf_up != 6:
                return (dE, dw,False)
            elif (conf_left !=2 and conf_left!=5):
                return (dE, dw,False)
            elif (conf_right !=1 and conf_right!=5):
                return (dE, dw, False)
            else:
                self.pattern[pos[0],pos[1],:] = conf3
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf4
                dE += 2 * (energymatrix[2] - energymatrix[4])
                dw *= (weightmatrix[2] / weightmatrix[4]) ** 2
                #moving the left white square
                if conf_left == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf6
                    dE += energymatrix[5] - energymatrix[1]
                    dw *= weightmatrix[5] / weightmatrix[1]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf1
                    dE += energymatrix[0] - energymatrix[4]
                    dw *= weightmatrix[0] / weightmatrix[4]
                #moving the right white square
                if conf_right == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf6
                    dE += energymatrix[5] - energymatrix[0]
                    dw *= weightmatrix[5] / weightmatrix[0]
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf2
                    dE += energymatrix[1] - energymatrix[4]
                    dw *= weightmatrix[1] / weightmatrix[4]
                return (dE, dw, True)

        return (dE, dw, False)


#    def local_update(self):
#        #introducing randomness
#        i = rnd.randint(0, self.m_trotter*self.n_spins)
#        i *= 2
#        #getting random position on the white squares
#        x = i // self.n_spins 
#        y = i % self.n_spins + x%2
#
#        self.local_update_pos(np.array([x,y], dtype = int))
#
#        return self.createimage()
    
    
    def local_update(self,):
        dE = 0
        dw = 1
        
        spinpos  = rnd.randint(0,self.n_spins)
        mpos  = 2*rnd.randint(0,self.m_trotter) + spinpos % 2
        pos = np.array([mpos,spinpos])
        i = mpos * self.n_spins + spinpos
        i_init = mpos * self.n_spins + spinpos
        dE,dw,has_changed = self.local_update_pos(pos)
        #print("try",pos,dE, dw)
        while (has_changed == False and i-i_init<self.n_spins*2*self.m_trotter ):
            i+=2
#            spinpos  = rnd.randint(0,self.n_spins)
#            mpos  = 2*rnd.randint(0,self.m_trotter) + spinpos % 2
            spinpos  = i%self.n_spins
            mpos  = (i//self.n_spins)%(2*self.m_trotter)
            if(spinpos + mpos % 2 ==1):
                spinpos += 1
                spinpos = spinpos%self.n_spins
                i+=1
            pos = np.array([mpos,spinpos])
            dE,dw,has_changed =self.local_update_pos(pos)
            #print("trylocal",pos,dE, dw, has_changed)
        return dE,dw  #has_changed
    
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
        
        dw = 1
        dE = 0
        test = self.copy()
        a = rnd.rand()
        b = rnd.rand()
        if (a<threshold):
            dEt,dwt= test.local_update()
            dE += dEt
            dw *= dwt
            mtype = "local"
        else:
            dEt,dwt= test.splitline()
            dE += dEt
            dw *= dwt
            mtype = "splitline"
        #print("try a = ",a,"dw = ",dw)
        if (dw>b):
            self.pattern = test.pattern
            print("change accepted"+mtype,dE, dw)
            return dE, dw
        print("aborted",mtype)
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
            
