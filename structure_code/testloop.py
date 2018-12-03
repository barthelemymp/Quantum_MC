import loopclass_2 as lp
import numpy as np
#import matplotlib.pyplot as plt

np.random.seed(2985729547)

s = "QMCMeanEnergyJxJzBeta0.1_1.txt"
with open(s, 'w') as fichier:
    for k in range (10):
        loop = lp.Loop_2(10, ( k + 1 ) * 0.01, 8, -1, -1)
        energ = loop.Quantum_Monte_Carlo()
        fichier.write("The mean energy for a periodic chain of length 8 with Jx, Jz, beta = -1, -1, " + str((k+1) * 0.01) + " is " + 
                      str(np.mean(energ)) + '+/-' + str(np.std(energ)/np.sqrt(len(energ))) + "\n")

#loop = lp.Loop_2(10,0.1,8,-2,-2)

##loop.spins = np.array([[1,0,1,0],
#                       [1,0,1,0],
#                       [1,0,1,0],
#                       [1,0,1,0],
#                       [1,0,1,0],
#                       [1,0,1,0],
#                       [1,0,1,0],
#                       [1,0,1,0]])

#loop.spins_to_pattern()
#print(loop.total_energy(), loop.weight())
#energ = loop.Quantum_Monte_Carlo()

#for k in range(10):
#    loop.spins_to_pattern()
#    loop.set_total_graph()
##    loop.creategraph()
#    loop.find_loops()
#    loop.createimage()


#for k in range(10):
#    loop.spins_to_pattern()
#    loop.set_total_graph()
#    loop.find_loops()
    
#loop.createimage()


