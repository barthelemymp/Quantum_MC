# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:43:42 2018

@author: etien
"""

import numpy as np
import pickle as pk
#import matplotlib.pyplot as plt


#fig = plt.figure(figsize = (15, 10))
x = np.arange(1, 20)
y = []
yerr = []

for k in range(2, 3):
    with open('QMCENERG_x=1z=05_beta1_m' + str(k) + '.txt', 'rb') as fichier:
        energy = pk.load(fichier)
        print(len(energy))
        print(str(np.mean(energy)) + '+/-' + str(np.std(energy)/np.sqrt(10000000)))
#    for l in x:
#        print(len(energy[::l]))
#        norm = np.sqrt(10000000. / l)
#        y += [np.mean(energy[::l])]
#        yerr += [np.std(energy[::l])/norm]
#ax1 = fig.add_subplot(121)      
#ax1.errorbar(x, y, yerr=yerr)
#ax2 = fig.add_subplot(122)
#ax2.plot(x, y)