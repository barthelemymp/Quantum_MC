# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 08:49:15 2018

@author: etien
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

with open('hundred_spins_energy.txt', 'rb') as fichier:
    energy = pk.load(fichier)
    
L = len(energy)
x = np.arange(1, L + 1)

fig = plt.figure(figsize = (15, 10))
energyhat = smooth(energy, 100)
plt.plot(x[100 : L - 100], energyhat[100 : L - 100], label = 'Energy')

plt.legend(loc = 0, fontsize = 'x-large')
plt.xlabel('Monte Carlo Moves', fontsize = 'xx-large')
plt.ylabel('$(Energy - 2.4999999e1) * 1e9$ ', fontsize = 'xx-large')

plt.title('Variation of the energy along the moves', fontsize = 'xx-large')

E = np.mean(energy)
Dev = np.std(energy)/np.sqrt(len(energy))
print(str(E) + ' +/- ' + str(Dev))

plt.savefig('Energy_hundred_spins.png')