# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:18:03 2018

@author: bmeyn
"""

import numpy as np
import numpy.random as rnd
import cv2

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

    def createimage(self, casesize=20):
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
        cv2.imshow('image', image)

        cv2.waitKey()
        
    
    
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
        #print("en",energy)
        return energy
    
    def weight(self):
        a = np.exp(self.dtau*self.Jz/4)
        cosh = np.cosh(self.dtau*self.Jx/2)
        sinh = np.sinh(self.dtau*self.Jx/2)
        weightmatrix = np.array([1/a, 1/a, -a*sinh, -a*sinh, a*cosh, a*cosh])
        weight = np.nanprod(self.pattern*weightmatrix)
        return weight
    
    def splitspin(self,pos,dE,dw): #probleme d'origine du dernier mouvement a regler 
        
        ae = self.Jz/4
        th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        energymatrix = np.array([-ae, -ae, ae+coth, ae+coth, ae+th, ae+th])
        
        aw = np.exp(self.dtau*self.Jz/4)
        cosh = np.cosh(self.dtau*self.Jx/2)
        sinh = np.sinh(self.dtau*self.Jx/2)
        weightmatrix = np.array([1/aw, 1/aw, -aw*sinh, -aw*sinh, aw*cosh, aw*cosh])
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
        return dE, dw
    
    def local_update_pos(self, pos): # ETIENNE RENVOIE DE ET DW
        """
        This method allows local updates, described in Fig.2 of the article. We will look for various type of 
        pattern, which are localised on four "white squares" in the pattern. We will call them the conf_down, 
        conf_up, conf_left, conf_right. 
        """
        ae = self.Jz/4
        th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        energymatrix = np.array([-ae, -ae, ae+coth, ae+coth, ae+th, ae+th])
        
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
            return 0
        #the bottom square is up-down==>down-up   
        elif conf_down == 3:
            #the up square must be down-up==>up-down
            if conf_up != 4:
                return 0
            #then we have two possibilities for the left square, either up-down==>up-down or down-down==>down-down
            elif (conf_left !=1 and conf_left!=6):
                return 0
            #then we have two possibilities for the right square, either up-down==>up-down or up-up==>up-up
            elif (conf_right !=2 and conf_right!=6):
                return 0
            #we move the spins from up-down==>down-up==>down-up==>up-down to 
            #up-down==>up-down==>up-down==>up-down as describe in the local update
            else:
                self.pattern[pos[0],pos[1],:] = conf6
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf6
                #moving the left white square
                if conf_left == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf5
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf2
                #moving the right white square
                if conf_right == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf5
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf1
                return 'changement d energie 3'
        #case in which the bottom square is down-up==>up-down
        elif conf_down == 4:
            if conf_up != 3:
                return 0
            elif (conf_left !=2 and conf_left!=5):
                return 0
            elif (conf_right !=1 and conf_right!=5):
                return 0
            else:
                self.pattern[pos[0],pos[1],:] = conf5
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf5
                #moving the left white square
                if conf_left == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf6
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf1
                #moving the right white square
                if conf_right == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf6
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf2
                return 'changement d energie 4'
        #case in which the bottom square is down-up==>down-up
        elif conf_down == 5:
            if conf_up != 5:
                return 0
            elif (conf_left !=1 and conf_left!=6):
                return 0
            elif (conf_right !=2 and conf_right!=6):
                return 0
            else:
                self.pattern[pos[0],pos[1],:] = conf4
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf3
                #moving the left white square
                if conf_left == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf5
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf2
                #moving the right white square
                if conf_right == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf5
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf1
                return 'changement d energie 5'
        #case in which the bottom square is updown==>up-down
        else:
            if conf_up != 6:
                return 0
            elif (conf_left !=2 and conf_left!=5):
                return 0
            elif (conf_right !=1 and conf_right!=5):
                return 0
            else:
                self.pattern[pos[0],pos[1],:] = conf3
                self.pattern[( pos[0] + 2 )%( 2 * self.m_trotter ),\
                             pos[1],\
                             :] = conf4
                #moving the left white square
                if conf_left == 2:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf6
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] - 1 )%self.n_spins,\
                                 :] = conf1
                #moving the right white square
                if conf_right == 1:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf6
                else:
                    self.pattern[( pos[0] + 1 )%( 2 * self.m_trotter ),\
                                 ( pos[1] + 1 )%self.n_spins,\
                                 :] = conf2
                return 'changement d energie 6'

        #il faut modifier la somme pour hanger conf left et conf right. 

        return 0
    
    
    def local_update(self,):
        mpos  = rnd.randint(0,2*self.m_trotter)
        spinpos  = rnd.randint(0,self.n_spins)
        pos = np.array([mpos,spinpos])
        has_changed = self.localupdate(pos)
        while (has_changed == 0):
            self.localupdate(pos)
        return has_changed
    
    def basic_move(self,n_splitline,n_localupdate):
        dw = 0
        dE = 0
        for i in range(n_splitline):
            dEtrans, dwtrans = self.splitline()
            dE += dEtrans
            dw *= dwtrans
        for j in range(n_localupdate):
            dEtrans, dwt = self.local_update()
            dE += dEtrans
            dw *= dwtrans
        return dE, dw
            
            
        
        
        
        
        
        
    
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

    
    
                
    
    
 
    







class config:
    """A Feynman path in position/imaginary time space"""

    def __init__(self, m_trotter, dtau, n_spins, Jx, Jz, mode = 'local', ):
        self.m_trotter = m_trotter #division of the temporary time"
        self.dtau = dtau
        self.n_spins = n_spins
        self.Jx = Jx
        self.Jz = Jz
        self.mode = mode


    def compute_decorrelation(self,n_splitline,n_localupdate):
        if(self.mode is 'local'):
            s = States(self.m_trotter,self.dtau,self.n_spins,self.Jx,self.Jz)
            for i in range(n_splitline):
                s.splitline()
            for i in range(n_localupdate):
                s.local_update()
                
            
            
        if(self.mode is 'loop'):
            return
                    
