import loopclass as lp
import numpy as np


np.random.seed(2985729547)

s = "QMCMeanEnergyJxJzBeta0.1_1.txt"
with open(s, 'w') as fichier:
    for k in range (2, 10):
        loop = lp.Loop(10, ( k + 1 ) * 0.01, 8, -1, -1)
        energ = loop.Quantum_Monte_Carlo()
        fichier.write("The mean energy for a periodic chain of length 8 with Jx, Jz, beta = -1, -1, " + str((k+1) * 0.01) + " is " + 
                      str(np.mean(energ)) + '+/-' + str(np.std(energ)/np.sqrt(len(energ))) + "\n")

