# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:18:03 2018

@author: bmeyn
"""

import numpy as np
import numpy.random as rnd
import cv2

class Loop:

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
        self.energymatrix = np.array([-self.a, -self.a, self.a+self.coth, self.a+self.coth, self.a+self.th, self.a+self.th])

        self.b = np.exp(self.dtau*self.Jz/4)
        self.cosh = np.cosh(self.dtau*self.Jx/2)
        self.sinh = np.sinh(self.dtau*self.Jx/2)
        self.weightmatrix = np.array([1/self.b, 1/self.b, -self.b*self.sinh, -self.b*self.sinh, self.b*self.cosh, self.b*self.cosh])

        self.greycase = np.ones((20,20),dtype=np.uint8) * 70
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
        self.cases = [self.case1, self.case2, self.case3, self.case4, self.case5, self.case6]

        self.inv = np.array([[ 0.5, -0.5,  0.5], \
                             [ 0.5,  0.5, -0.5], \
                             [-0.5,  0.5,  0.5]])
        self.w   = np.array([self.b*self.cosh, self.b*self.sinh, 1/self.b])
        self.w11 = np.dot(self.inv, self.w)[0]
        self.w12 = np.dot(self.inv, self.w)[1]
        self.w24 = np.dot(self.inv, self.w)[2]
        self.w22 = np.dot(self.inv, self.w)[1]
        self.w31 = np.dot(self.inv, self.w)[0]
        self.w34 = np.dot(self.inv, self.w)[2]

        self.graph1 = self.case2
        self.graph2 = np.transpose(self.case2)
        self.graph3 = np.ones((20,20))*255
        self.graph4 = self.case3 + self.case4
        self.graphs = [self.graph1, self.graph2, self.graph3, self.graph4]

    def createimage(self):       
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
                if((i+j+1)%2):
                    conf = np.nanargmax(np.array(self.pattern[i,j,:]))
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=self.cases[conf]
                else:
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=130
                    
                
        image = np.array(image,dtype=np.uint8)
        cv2.imshow('image', image)

    def total_energy(self):
        energy = np.nansum(self.pattern*self.energymatrix)
        return energy
    
    def weight(self):
        weight = np.nanprod(self.pattern*self.weightmatrix)
        return weight



    def white_squareingraph(self, pos):
        conf = np.nanargmax(np.array(self.pattern[pos[0],pos[1],:]))

        #image_conf = self.cases[conf]

        #cv2.imshow('conf', image_conf)

        if conf < 2:
            prob = self.w31/self.weightmatrix[0]
            if rnd.random() < prob :
                graph = 1
            else:
                graph = 4
        elif conf < 4:
            prob = self.w22/self.weightmatrix[2]
            if rnd.random() < prob :
                graph = 2
            else:
                graph = 4
        else:
            prob = self.w11/self.weightmatrix[4]
            if rnd.random() < prob :
                graph = 1
            else:
                graph = 2

        image_graph = self.graphs[graph - 1]

        return image_graph

    def creategraph(self):
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
                if((i+j+1)%2):
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=self.white_squareingraph([i, j])
                else:
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=130
                    
                
        image = np.array(image,dtype=np.uint8)
        cv2.imshow('graphs', image)
