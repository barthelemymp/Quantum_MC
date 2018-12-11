# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:00:28 2018

@author: etien
"""

import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

#initializing the figure
fig = plt.figure(figsize = (15,10))

#delta_tau values
x = 1 / np.arange(2, 8)
x_pol = np.linspace(0, 0.5, 50)

#Theoretical values given by M.Ferrero
y_th = [-0.9014675456746862, -0.8812701667881313, -0.8741886937808667,
        -0.8709103141965382, -0.8691294502607925, -0.8680556681106003]
y = []
yerr = []




#results computed with the following parameters from loopclass
# Jx = Jz = 1
# beta = 1
# m 2 --> 7
# n_cycles = 1e7
# length_cycle = 1
for k in range(2, 8):
    with open('QMCENERG_x=z_beta1_m' + str(k) + '.txt', 'rb') as fichier:
        norm = np.sqrt(10000000)
        energy = pk.load(fichier)
        y += [np.mean(energy)]
        yerr += [np.std(energy)/norm]

#polynomial fit
coef = np.polyfit(x, y_th, 3)
y_th_pol = np.polyval(coef, x_pol)

#editing        
plt.xlabel('delta_tau', fontsize = 'xx-large')
plt.ylabel('energy', fontsize = 'xx-large')
plt.xticks(fontsize = 'x-large')
plt.yticks(fontsize = 'x-large')
plt.title('Theoretical Energy and Computed energy, \n beta = 1, Jx = Jz = 1', fontsize = 'xx-large')

#plt.errorbar(x, y, yerr = yerr, label = 'Value obtained by the loop algorithm \n with their errobars',
#             lw = 4, alpha = 0.7)
plt.plot(x, y_th, '--', label = 'Theoretical values for delta_tau',
          lw = 5, alpha = 0.7)
plt.plot(x_pol, y_th_pol, ':', label = 'Theoretical values for delta_tau',
          lw = 5, alpha = 0.7)
plt.plot([0, 1/2],[-0.8650823114175719, -0.8650823114175719], label = 'Exact value')

plt.legend(loc = 0, fontsize = 'x-large')

#print(y_pol[0])

#saving
#plt.savefig('ThvsComp_x=z=1_m2-8.png')
