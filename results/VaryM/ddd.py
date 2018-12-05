# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:43:42 2018

@author: etien
"""

import numpy as np
import pickle as pk
import matplotlib.pyplot as plt


fig = plt.figure(figsize = (15, 10))
x = np.arange(1, 100)
y = []
yerr = []

with open('QMCENERG_x=y_beta1_m' + str(4) + '.txt', 'rb') as fichier:
    energy = pk.load(fichier)
    for k in x:
#        print(len(energy[::k]))
        norm = np.sqrt(10000000. / k)
        y += [np.mean(energy[::k])]
        yerr += [np.std(energy[::k])/norm]
ax1 = fig.add_subplot(121)      
ax1.errorbar(x, y, yerr=yerr)
ax2 = fig.add_subplot(122)
ax2.plot(x, y)