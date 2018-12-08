# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:43:42 2018

@author: etien
"""


import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

#initializing the figure
fig = plt.figure(figsize = (15,10))

#delta_tau values
x = 1 / np.arange(2, 8)

#Theoretical values given by M.Ferrero
y_th = [-0.644040296998033, -0.6353376460680784, -0.6322837584550399,
        -0.630869118123033, -0.6301004016764887, -0.6296368031123277]

y = []
yerr = []

#results computed with the following parameters from loopclass
# Jx = Jz = 1
# beta = 1
# m 2 --> 7
# n_cycles = 1e7
# length_cycle = 1
for k in range(2, 8):
    with open('QMCENERG_x=1z=05_beta1_m' + str(k) + '.txt', 'rb') as fichier:
        norm = np.sqrt(10000000)
        energy = pk.load(fichier)
        y += [np.mean(energy)]
        yerr += [np.std(energy)/norm] 


#editing        
plt.xlabel('delta_tau', fontsize = 'xx-large')
plt.ylabel('energy', fontsize = 'xx-large')
plt.xticks(fontsize = 'x-large')
plt.yticks(fontsize = 'x-large')
plt.title('Theoretical Energy and Computed energy, \n \beta = 1, $J_{x}$ = 1, $J_z$ = 0.5', fontsize = 'xx-large')

plt.errorbar(x, y, yerr = yerr, label = 'Value obtained by the loop algorithm \n with their errobars',
             lw = 4, alpha = 0.7)
plt.plot(x, y_th, '--', label = 'Theoretical values',
          lw = 5, alpha = 0.7)

plt.legend(loc = 0, fontsize = 'x-large')

#saving
plt.savefig('ThvsComp_x=1z=05_m2-7.png')