import loopclass_2 as lp
import numpy as np
#import matplotlib.pyplot as plt

#np.random.seed(1)

loop = lp.Loop_2(4,1,4,1,-1)

loop.spins = np.array([[1,0,0,0],
                       [1,0,0,0],
                       [1,0,0,0],
                       [1,0,0,0],
                       [1,0,0,0],
                       [1,0,0,0],
                       [1,0,0,0],
                       [1,0,0,0]])

loop.spins_to_pattern()
print(loop.total_energy(), loop.weight())
#energ = loop.Quantum_Monte_Carlo()

#for k in range(10):
#    loop.spins_to_pattern()
#    loop.set_total_graph()
#    loop.find_loops()
#    if k > 7 : 
#        print(loop.spins)
#        loop.createimage()
#loop.createimage()
#loop.spins_to_pattern()
#loop.createimage()


#for k in range(10):
#    loop.spins_to_pattern()
#    loop.set_total_graph()
#    loop.find_loops()
    
#loop.createimage()


