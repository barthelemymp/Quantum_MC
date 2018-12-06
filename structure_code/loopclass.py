# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:12:45 2018

@author: etien
"""


import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

class Loop:

    def __init__(self, m_trotter, dtau, n_spins, Jx, Jz):
        """
        Constructing a 2D pattern with beta = m_trotter * dtau
       The horizontal dimension is the spatial one, with n_spins steps
       The vertical one is the temporal one, with 2*m_trotter steps
       The interactions are defined by Jx and Jz
        """

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
        self.pattern = np.zeros((2*m_trotter,n_spins), dtype = int)
        
        self.p_right = np.diag(np.ones(self.n_spins, dtype = int)) + np.diag(2 * np.ones(self.n_spins - 1, dtype = int), k = -1) + np.diag([2], k = self.n_spins - 1)
        self.p_left = np.diag(3 * np.ones(2 * self.m_trotter, dtype = int)) + np.diag(np.ones(2 * self.m_trotter - 1, dtype = int), k = 1) + np.diag([1], k = -2 * self.m_trotter + 1)
        
        self.p_mask = np.zeros((2*self.m_trotter, self.n_spins), dtype = int)
        for i in range(2 * self.m_trotter):
            for j in range(self.n_spins):
                if (i + j + 1)%2:
                    self.p_mask[i, j] = 1
        self.p_mask = self.p_mask.astype(bool)
        
        #in the loop algorithm, we need to have a representation of the spins in a graph
        self.total_graph = 5 * np.ones((2*self.m_trotter, self.n_spins), dtype = int)
        
        
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
        self.weightmatrix = np.array([1/self.b, 1, 1, 1,
                                      self.b*self.cosh, self.b*self.sinh, 1,
                                      self.b*self.sinh, self.b*self.cosh, 
                                      1, 1, 1, 1/self.b])
        
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
        
        #computing the weight for the passage from the pattern to the graph
        #the graph 3 (cf Article, i.e. the graph along witch each spin is flipped)
        #is NOT allowed
        #The weight of each graph depending on the previous configuration of
        #the pattern is computed thanks to equation (45) from the Article
        self.inv = 0.5 * np.array([[ 1., -1.,  1.], \
                                   [ 1.,  1., -1.], \
                                   [ -1.,  1.,  1.]])
        self.w   = np.array([self.b*self.cosh, self.b*self.sinh, 1/self.b])
        self.w11 = np.dot(self.inv, self.w)[0]
        self.w12 = np.dot(self.inv, self.w)[1]
        self.w24 = np.dot(self.inv, self.w)[2]
        self.w22 = np.dot(self.inv, self.w)[1]
        self.w31 = np.dot(self.inv, self.w)[0]
        self.w34 = self.w24
#        self.w11 = 0.5
#        self.w12 = 0.5
#        self.w24 = 0.5
#        self.w22 = 0.5
#        self.w31 = 0.5
#        self.w34 = 0.5
        
        #initializing the graphs
        self.graph1 = self.case2
        self.graph2 = np.transpose(self.case2)
        self.graph3 = np.ones((20,20), dtype = np.uint8)*255
        self.graph4 = np.ones((20,20))*255
        for i in range(19):
            self.graph4[i,19-i]=0
            self.graph4[i,18-i]=0
            self.graph4[i,i]=0
            self.graph4[i,i+1]=0
        self.graphs = [self.graph1, self.graph2, self.graph3, self.graph4, self.greycase]


    def total_energy(self):
        """
        Computes the Energy of the configuration. Uses self.pattern to know the tiles
        Then uses self.energymatrix to know the energy of each tile. Sum over them.
        """
        pattern_energy = self.energymatrix[self.pattern]
#        print(pattern_energy)
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
#        fig, ax = plt.subplots(figsize = (10,10))
        
        #this array corresponds to the image
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
                if((i+j+1)%2):
                    tile = self.pattern[i, j]
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=self.cases[tile]
                else:
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=130
        
        image = np.array(image)
        
        return image
#        ax.imshow(image, cmap = "Greys_r")

                
    def tile_in_graph(self, pos):
        """
        This method is the core of the loop algorithm, turning the "pattern" representation
        into the "graph" one. Here, for a specified tle, we get an adapted graph with
        respect to the weight defined in the article and computed in self.wIJ with I : tile and J : graph
        """
        graph = 5
        
        #first, let us be sure that the considered tile is white. If not, we name
        #the "grey" graph 5.
        tile = [1, 0, 0, 0, 6, 3, 0, 4, 5, 0, 0, 0, 2][self.pattern[pos[0], pos[1]]]
#        print(tile)
        if not tile:
            return graph
        
        #now, we are sure that at least one element of conf_array is not nan
        #we get the tile
#        tile = np.nanargmax(tile_array)

        #going over the possible choices
        if tile < 3:     #the tile has a weight w[3]. So the graph is 1 or 4
            #the probality of choosing each and the choice according to it
            graph = 1
                
        elif 2 < tile < 5:   #the tile is of type 2. So the graph is 2 or 4
            prob = self.w22/self.w[1]
            if rnd.random() < prob :
                graph = 2
            else:
                graph = 4
                
        else:            #the tile is of type 1. So the graph is 1 or 2
            prob = self.w11/self.w[0]
            if rnd.random() < prob :
                graph = 1
            else:
                graph = 2
        
        #we have chosen the graph
        return graph
    
    def set_total_graph(self):
        """
        Using the tile_in_graph method, we compute the whole graph
        """
        for i in range(2*self.m_trotter):
            for j in range(self.n_spins):
                self.total_graph[i,j] = self.tile_in_graph(np.array([i, j]))
        return
        
    
    def find_next(self, spin, pos_graph):
        """
        In order to find the loops, we go over total_graph. Given one spin and 
        one associated graph, we are able to know the next spin in the total_graph
        and the next graph, which are returned by this method
        """
        
        #extract the graph
        graph = self.total_graph[pos_graph]
        #let us be sure that our method is coherent and does not fall on a grey graph
        assert(graph != 5)
        
        #Parameters to compute easily
        modulo_trotter = 2*self.m_trotter
        modulo_spin    = self.n_spins
        
        #given the spin position "spin", we compute the other 
        #accessible spin positions
        spin_i_minus = (spin[0] - 1)%modulo_trotter
        spin_i       =  spin[0]
        spin_i_plus  = (spin[0] + 1)%modulo_trotter
        spin_j_minus = (spin[1] - 1)%modulo_spin
        spin_j       =  spin[1]
        spin_j_plus  = (spin[1] + 1)%modulo_spin
        
        #given the graph position "pos_graph", we compute the other 
        #accessible graph positions
        pos_graph_up    = (pos_graph[0] + 1)%modulo_trotter
        pos_graph_down  = (pos_graph[0] - 1)%modulo_trotter
        pos_graph_left  = (pos_graph[1] - 1)%modulo_spin
        pos_graph_right = (pos_graph[1] + 1)%modulo_spin
        
        #where is the spin on the graph. We need to discrimine on the corner on 
        if spin_i == pos_graph[0]:
            #spin is on the bottom of the graph
            if spin_j == pos_graph[1]:
                #spin is bottom left of the graph
                if graph == 1:
                    #graph is vertical
                    return ((spin_i_plus, spin_j), (pos_graph_up, pos_graph_left))
                elif graph == 2:
                    #graph is horizontal
                    return ((spin_i, spin_j_plus), (pos_graph_down, pos_graph_right))
                elif graph == 4:
                    #graph is crossed
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
                    #graph is crossed
                    return ((spin_i_plus, spin_j_minus), (pos_graph_up, pos_graph_left))
        else:
            #spin is on the top of the graph
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
                    return ((spin_i_minus, spin_j_minus), (pos_graph_down, pos_graph_left))
        
        return
    
    
    def find_loops(self):
        """
        This method find the loops on the given graph and flip the spins along them
        with probability 0.5
        """
        
        #Need to know whether a spin is already in the loop or not, thanks
        #to the "done" list
        done = []
        
        #Let us pass the loops in class parameter in order to be able to access them
        self.loops = []
        
        #Going over all the spins
        for i in range(2*self.m_trotter):
            for j in range(self.n_spins):
                
                #Is the spin already considered ? 
                if not ((i, j) in done):
                    
                    #boolean probability to flip the total loop
                    bool_prob = rnd.random() < 0.5
                    if bool_prob:
                        #let flip the spin
                        self.spins[i, j] = int(not self.spins[i, j])
                        
                    #actualize the done list
                    done += [(i,j)]
                    
                    #create new loop
                    new_loop = [(i, j)]
                    
                    #we now need to find the whole loop. Let us get the next spin and 
                    #loop until we find back the first spin.
                    temp = self.find_next((i,j), (2*(i//2),2*(j//2)))
                    
                    #loop. temp is of the form (position of the spin, position of the graph)
                    while (temp[0] != (i,j)):
                        indexes = temp[0]
                        #actualize the loop
                        new_loop += [indexes]
                        #the spin will become "done"
                        done += [indexes]
                        #next step
                        temp = self.find_next(indexes, temp[1])
                        
                        #flip the spin
                        if bool_prob:
                            self.spins[indexes[0], indexes[1]] = int(not self.spins[indexes[0], indexes[1]])
                    
                    #Store the loops in case
                    self.loops += [new_loop]
    

    def creategraph(self):
        """
        Give the graph representation of the configuration
        """
        
        #initializing
#        fig, ax = plt.subplots(figsize = (10, 10))
        
        #this array is the image
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
                if((i+j+1)%2):
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=self.graphs[self.total_graph[i, j] - 1]
                else:
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=130  
                    
        image = np.array(image)
        return image
#        ax.imshow(image, cmap = 'Greys_r')
    
    
    def pattern_to_string(self):
        st = ''
        for i in range(2 * self.m_trotter):
            for j in range(self.n_spins):
                st += '1' if self.spins[i, j] else '0'
        return st
    
    def QMC_step(self):
        self.spins_to_pattern()
        self.set_total_graph()
        self.find_loops()
           
    def Quantum_Monte_Carlo(self, n_warmup=100, n_cycles = 10000, length_cycle = 1):
        
#        pattern_done = {}
        energ = np.zeros(n_cycles)
        # Monte Carlo simulation
        for n in range(n_warmup+n_cycles):
            print(n)
            # Monte Carlo moves
            for l in range(length_cycle):
#                pattern_done[self.pattern_to_string()] = 1
                self.spins_to_pattern()
                self.set_total_graph()
                self.find_loops()
            # measures
            if n >= n_warmup:
                e = self.total_energy()
#                if e > 0: break
                energ[n-n_warmup] = e
#                print("ener",e)
#        print('Energy:', np.mean(energ), '+/-', np.std(energ)/np.sqrt(len(energ)))
        return energ
