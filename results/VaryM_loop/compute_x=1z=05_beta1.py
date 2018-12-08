# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:19:27 2018

@author: etien

This script allows to compute the files .txt of the same folder
"""

import loopclass as lp
import numpy as np
import pickle as pk

#setting the seed
np.random.seed(299792458)

#Parameters : Jx = 1. Jz = 0.5
# beta = 1
# m 2 -- > 7
# n_cycles = 1e7
# length cycle = 1
def QMC_mean_energy(k):
    with open('QMCENERG_x=1z=05_beta1_m' + str(k) + '.txt', 'wb') as fichier:
        loop = lp.Loop(k, 1./ k, 4, 1., 0.5)
        energ = loop.Quantum_Monte_Carlo(n_cycles = 10000000)
        pk.dump(energ, fichier)

for k in range(2, 8):
    QMC_mean_energy(k) 