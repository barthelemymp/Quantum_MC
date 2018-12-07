import loopclass as lp
import numpy as np


np.random.seed(2985729547)

s = "QMCMeanEnergyJx1Jz0.5Beta0.1_1.txt"
with open(s, 'w') as fichier:
    for k in range (1, 11):
        loop = lp.Loop(10, ( k + 1 ) * 0.1, 8, 1, 0.5)
        energ = loop.Quantum_Monte_Carlo()
        fichier.write("The mean energy for a periodic chain of length 8 with Jx, Jz, beta = 1, 0.5, " + str((k+1) * 0.1) + " is " + 
                      str(np.mean(energ)) + '+/-' + str(np.std(energ)/np.sqrt(len(energ))) + "\n")

