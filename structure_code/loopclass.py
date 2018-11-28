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
        """Constructing a 2D pattern with beta = m_trotter * dtau
           The horizontal dimension is the spatial one, with n_spins steps
           the vertical one is the temporal one, with 2*m_trotter steps
           The interaction are defined by Jx and Jz
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
        self.spins = np.zeros((2*m_trotter, n_spins))
        
        #this matrix allows the computation of the energy and the weight 
        #it also allows the representation of the pattern given in the
        #Article (with a grey and white board, each black line is a line
        #of up spins at the corner of the tiles)
        self.pattern = np.zeros((2*m_trotter,n_spins,6))
        self.pattern[:,:,:]=np.nan
        
        #in the loop algorithm, we need to have a representation of the spins in a graph
        self.total_graph = np.zeros((2*self.m_trotter, self.n_spins), dtype = int)
        
        
        #computing the energy depending on the configuration of the tiles.
        #Each tile has a particular energy, and one has to sum over the white tiles 
        #to find the energy. This is why we adopted the "pattern" representation
        self.a = self.Jz/4
        self.th = self.Jx/2*np.tanh(self.dtau*self.Jx/2)
        self.coth = self.Jx/(2*np.tanh(self.dtau*self.Jx/2))
        self.energymatrix = np.array([-self.a, -self.a, self.a+self.coth, \
                                      self.a+self.coth, self.a+self.th, self.a+self.th])
    
        #computing the weight depending on the configuration of the tiles. 
        #Each tile has a particular weight, and one has to make the product over the white tiles 
        #to find the weight.
        self.b = np.exp(self.dtau*self.Jz/4)
        self.cosh = np.cosh(self.dtau*self.Jx/2)
        self.sinh = np.sinh(self.dtau*self.Jx/2)
        self.weightmatrix = np.array([1/self.b, 1/self.b, -self.b*self.sinh,\
                                      -self.b*self.sinh, self.b*self.cosh, self.b*self.cosh])
        
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
        
        #initializing the graphs
        self.graph1 = self.case2
        self.graph2 = np.transpose(self.case2)
        self.graph3 = np.ones((20,20))*255
        self.graph4 = self.case3/2 + self.case4/2
        self.graphs = [self.graph1, self.graph2, self.graph3, self.graph4, self.greycase]


    def total_energy(self):
        """
        Computes the Energy of the configuration. Uses self.pattern to know the tiles
        Then uses self.energymatrix to know the energy of each tile. Sum over them.
        """
        energy = np.nansum(self.pattern*self.energymatrix)
        return energy
    
    
    def weight(self):
        """
        Computes the Weight of the configuration. Uses self.pattern to know the tiles
        Then uses self.weightmatrix to know the weight of each tile. Make the product of them.
        """
        weight = np.nanprod(self.pattern*self.weightmatrix)
        return weight

    
    def spins_to_pattern(self):
        """
        Given the spin configuration, turn it into a pattern configuration. Allows the
        image to be created or the graph to be computed.
        """
        
        #going over all the tiles
        for l in range(self.m_trotter):
            for j in range(self.n_spins):
                
                #initialize the tile
                tile = np.zeros(6)
                tile[:]=np.nan
                
                #black and white board, and choosing only white tiles
                if j % 2:
                    i = 2 * l + 1
                else:
                    i = 2 * l 
                
                #how are the spins evolving along each tiles. Each boolean tells us about
                #if the concerned spin are up or not.
                vertg = np.bool(self.spins[i, j] * self.spins[(i+1)%(2*self.m_trotter), j])
                vertd = np.bool(self.spins[i, (j+1)%(self.n_spins)] * self.spins[(i+1)%(2*self.m_trotter), (j+1)%(self.n_spins)])
                diag1 = np.bool(self.spins[i, j] * self.spins[(i+1)%(2*self.m_trotter), (j+1)%(self.n_spins)])
                diag2 = np.bool(self.spins[i, (j+1)%(self.n_spins)] * self.spins[(i+1)%(2*self.m_trotter), j])

                #going over the possible tiles
                
                if vertg*vertd:   #all spins are up
                    tile[1] = 1
                elif diag1:       #only the first diagonal
                    tile[2] = 1
                elif diag2:       #only the second one
                    tile[3] = 1
                elif vertg:       #only the left vertical
                    tile[5] = 1
                elif vertd:       #only the right vertical
                    tile[4] = 1
                else:             #non of them are up
                    tile[0] = 1
                
                #at this point, the tile has been chosen
                self.pattern[i, j] = tile
        return
    
    
    def plot_image(self):
        """
        
        """
        x, y = [], []
        for i in range(2*self.m_trotter):
            for j in range(self.n_spins):
                if self.spins[i,j]:
                    x += [i]
                    y += [j]
        plt.scatter(y,x)
    
    def createimage(self):
        """
        Give the pattern representation of the configuration on the screen
        """
        
        #initializing the figure
        fig, ax = plt.subplots(figsize = (10,10))
        
        #this array corresponds to the image
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
                if((i+j+1)%2):
                    tile = np.nanargmax(np.array(self.pattern[i,j,:]))
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=self.cases[tile]
                else:
                    image[20*(l-1):20*(l),20*j:20*(j+1)]=130
        
        image = np.array(image,dtype=np.uint8)
        ax.imshow(image, cmap = "Greys_r")

                
    def tile_in_graph(self, pos):
        """
        This method is the core of the loop algorithm, turning the "pattern" representation
        into the "graph" one. Here, for a specified tle, we get an adapted graph with
        respect to the weight defined in the article and computed in self.wIJ with I : tile and J : graph
        """
        
        #first, let us be sure that the considered tile is white. If not, we name
        #the "grey" graph 5.
        tile_array = np.array(self.pattern[pos[0],pos[1],:])
        if not (False in np.isnan(tile_array)):
            return 5
        
        #now, we are sure that at least one element of conf_array is not nan
        #we get the tile
        tile = np.nanargmax(tile_array)

        #going over the possible choices
        if tile < 2:     #the tile is of type 3. So the graph is 1 or 4
            #the probality of choosing each and the choice according to it
            prob = self.w31/self.weightmatrix[0]
            if rnd.random() < prob :
                graph = 1
            else:
                graph = 4
                
        elif tile < 4:   #the tile is of type 2. So the graph is 2 or 4
            prob = self.w22/self.weightmatrix[2]
            if rnd.random() < prob :
                graph = 2
            else:
                graph = 4
                
        else:            #the tile is of type 1. So the graph is 1 or 2
            prob = self.w11/self.weightmatrix[4]
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
                    return ((spin_i_minus, spin_j_minus), (pos_graph_down, pos_graph_left))
        return
    
    def find_loops(self):
        done = []
        self.loops = []
        for i in range(2*self.m_trotter):
            for j in range(self.n_spins):
                if not ((i, j) in done):
                    prob = rnd.random()
                    bool_prob = prob < 0.5
#                    print(bool_prob)
                    if bool_prob:
                        self.spins[i, j] = int(not self.spins[i, j])
                    done += [(i,j)]
#                    k = 0
                    new_loop = [(i, j)]
#                    print(i, j, i//2, j//2)
                    temp = self.find_next((i,j), (2*(i//2),2*(j//2)))
#                    print(temp)
                    while (temp[0] != (i,j)):
                        indexes = temp[0]
                        new_loop += [indexes]
                        done += [indexes]
                        temp = self.find_next(indexes, temp[1])
#                        print(temp[0])
                        
#                        print('new loop', new_loop)
                        if bool_prob:
                            self.spins[indexes[0], indexes[1]] = int(not self.spins[indexes[0], indexes[1]])

#                        k += 1
                    self.loops += [new_loop]
#                    print(bool_prob, new_loop)
#                    print(self.spins)
    
    
    def spin_flip_along_loop(self, ):
        return
        
    
    

    def creategraph(self):
        fig, ax = plt.subplots(figsize = (10, 10))
        image = np.zeros((20*self.m_trotter*2,20*self.n_spins))
        for i in range(self.m_trotter*2):
            l = self.m_trotter*2 - i
            
            for j in range(self.n_spins):
#                print(self.tile_in_graph([i, j]))
                image[20*(l-1):20*(l),20*j:20*(j+1)]=self.graphs[self.total_graph[i,j] - 1]
                    
#        print(image)
        image = np.array(image,dtype=np.uint8)
        ax.imshow(image, cmap = 'Greys_r')
        return
    
    
    def Quantum_Monte_Carlo(self,n_warmup=100,n_cycles = 200,length_cycle = 100):
        
        energ = np.zeros(n_cycles)
        # Monte Carlo simulation
        for n in range(n_warmup+n_cycles):
            print(n)
            # Monte Carlo moves
            for l in range(length_cycle):
                self.spins_to_pattern()
                self.set_total_graph()
                self.find_loops()
            # measures
            if n >= n_warmup:
                e = self.total_energy()
                energ[n-n_warmup] = e
                print("ener",e)
        print('Energy:', np.mean(energ), '+/-', np.std(energ)/np.sqrt(len(energ)))
        return energ
