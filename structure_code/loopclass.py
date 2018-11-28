# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:18:03 2018

@author: bmeyn
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

class Loop:

    def __init__(self, m_trotter, dtau, n_spins, Jx, Jz):
        """Construct a path at inverse temperature beta in dim spatial
           dimensions and ntau imaginary time points"""

        #division of the imagingary time
        self.m_trotter = m_trotter 
        self.dtau = dtau
        #number of spins along the chain
        self.n_spins = n_spins
        #interactions
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
        #graph
        self.total_graph = np.zeros((2*self.m_trotter, self.n_spins), dtype = int)
        #spins
        self.spins = np.zeros((2*m_trotter, n_spins))
        
        #computing the energy depending on the configuration of the white
        #squares. cf Article
        self.a = self.Jz/4
        self.th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        self.coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        self.energymatrix = np.array([-self.a, -self.a, self.a+self.coth, self.a+self.coth, self.a+self.th, self.a+self.th])
        #computing the weight depending on the configuration of the white
        #squares. cf Article
        self.b = np.exp(self.dtau*self.Jz/4)
        self.cosh = np.cosh(self.dtau*self.Jx/2)
        self.sinh = np.sinh(self.dtau*self.Jx/2)
        self.weightmatrix = np.array([1/self.b, 1/self.b, -self.b*self.sinh, -self.b*self.sinh, self.b*self.cosh, self.b*self.cosh])
        
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
        self.cases = [self.case1, self.case2, self.case3, self.case4, self.case5, self.case6]
        
        #computing the weight for the passage from the pattern to the graph
        #the graph 3 (cf Article, i.e. the graph along witch each spin is flipped)
        #is NOT allowed
        #The weight of each graph depending on the previous configuration of
        #the pattern is computed thanks to equation (45) from the Article
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
        #initilizing the graphs
        self.graph1 = self.case2
        self.graph2 = np.transpose(self.case2)
        self.graph3 = np.ones((20,20))*255
        self.graph4 = self.case3/2 + self.case4/2
        self.graphs = [self.graph1, self.graph2, self.graph3, self.graph4, self.greycase]


    def total_energy(self):
        energy = np.nansum(self.pattern*self.energymatrix)
        return energy
    
    def weight(self):
        weight = np.nanprod(self.pattern*self.weightmatrix)
        return weight

    
    def spins_to_pattern(self):
        for i in range(self.n_spins):
            for l in range(self.m_trotter):
                conf = np.zeros(6)
                conf[:]=np.nan
                if i % 2:
                    j = 2 * l + 1
                else:
                    j = 2 * l 
                vertg = np.bool(self.spins[i, j] * self.spins[(i+1)%self.n_spins, j])
                vertd = np.bool(self.spins[i, (j+1)%(2*self.m_trotter)] * self.spins[(i+1)%self.n_spins, (j+1)%(2*self.m_trotter)])
                diag1 = np.bool(self.spins[i, j] * self.spins[(i+1)%self.n_spins, (j+1)%(2*self.m_trotter)])
                diag2 = np.bool(self.spins[i, (j+1)%(2*self.m_trotter)] * self.spins[(i+1)%self.n_spins, j])
#                print(i, j, vertg, vertd, diag1, diag2)
                if vertg*vertd:
                    assert(diag1)
                    assert(diag2)
                    conf[1] = 1
                elif diag1:
                    conf[2] = 1
                elif diag2:
                    conf[3] = 1
                elif vertg:
                    conf[5] = 1
                elif vertd:
                    conf[4] = 1
                else:
                    conf[0] = 1
                
                self.pattern[i, j] = conf
        return
    
    def createimage(self):    
        fig, ax = plt.subplots(figsize = (10,10))
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
        ax.imshow(image, cmap = "Greys_r")

                
    def white_square_in_graph(self, pos):
        conf_array = np.array(self.pattern[pos[0],pos[1],:])
        if not (False in np.isnan(conf_array)):
            return 5
        
        conf = np.nanargmax(conf_array)

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

        return graph
    
    def set_total_graph(self):
        for i in range(self.n_spins):
            for j in range(2*self.m_trotter):
                self.total_graph[i,j] = self.white_square_in_graph(np.array([i, j]))
        return
        
    
    def loop_from_graph(self, spin, pos_graph):
        graph = self.total_graph[pos_graph]
        assert(graph != 5)
        
        modulo_trotter = 2*self.m_trotter
        modulo_spin = self.n_spins
        
        spin_i_minus = (spin[0] - 1)%modulo_trotter
        spin_i = spin[0]
        spin_i_plus = (spin[0] + 1)%modulo_trotter
        spin_j_minus = (spin[1] - 1)%modulo_spin
        spin_j = spin[1]
        spin_j_plus = (spin[1] + 1)%modulo_spin
        
        pos_graph_up = (pos_graph[0] + 1)%modulo_trotter
        pos_graph_down = (pos_graph[0] - 1)%modulo_trotter
        pos_graph_left = (pos_graph[1] - 1)%modulo_spin
        pos_graph_right = (pos_graph[1] + 1)%modulo_spin
        
        #where is the spin on the graph ?
        if spin_i == pos_graph[0]:
            if spin_j == pos_graph[1]:
                #spin is bottom left of the graph
                if graph == 1:
                    #graph is vertical
                    return ((spin_i_plus, spin_j), (pos_graph_up, pos_graph_left))
                elif graph == 2:
                    #graph is horizontal
                    return ((spin_i, spin_j_plus), (pos_graph_down, pos_graph_right))
                elif graph == 4:
                    #graph is cross
                    return ((spin_i_plus, spin_j_plus), (pos_graph_up, pos_graph_right))
            else:
                #spin is bottom right
                if graph == 1:
                    #graph is vertical
                    return ((spin_i_plus, spin_j), (pos_graph_up, pos_graph_right))
                elif graph == 2:
                    #graph is horizontal
                    return ((spin_i, spin_j_minus), (pos_graph_down, pos_graph_left))
                elif graph == 4:
                    #graph is cross
                    return ((spin_i_plus, spin_j_minus), (pos_graph_up, pos_graph_left))
        else:
            if spin[1] == pos_graph[1]:
                #spin is up left
                if graph == 1:
                    #graph is vertical
                    return ((spin_i_minus, spin_j), (pos_graph_down, pos_graph_left))
                elif graph == 2:
                    #graph is horizontal
                    return ((spin_i, spin_j_plus), (pos_graph_up, pos_graph_right))
                elif graph == 4:
                    #graph is cross
                    return ((spin_i_minus, spin_j_plus), (pos_graph_down, pos_graph_right))
            else:
                #spin is up right
                if graph == 1:
                    #graph is vertical
                    return ((spin_i_minus, spin_j), (pos_graph_down, pos_graph_right))
                elif graph == 2:
                    #graph is horizontal
                    return ((spin_i, spin_j_minus), (pos_graph_up, pos_graph_left))
                elif graph == 4:
                    #graph is cross
                    return ((spin_i_minus, spin_j_minus), (pos_graph_down, pos_graph_up))
        return
    
    
    
    def spin_flip_along_loop(self, ):
        return
        
    
    

    def creategraph(self):
        fig, ax = plt.subplots(figsize = (10, 10))
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
#                print(self.white_square_in_graph([i, j]))
                image[20*(l-1):20*(l),20*j:20*(j+1)]=self.graphs[self.total_graph[i,j] - 1]
                    
#        print(image)
        image = np.array(image,dtype=np.uint8)
        ax.imshow(image, cmap = 'Greys_r')
