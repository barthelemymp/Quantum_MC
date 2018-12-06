import loopclass as lp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#np.random.seed(2985729547)
"""
s = "QMCMeanEnergy
with open(s, 'w') as fichier:
    for k in range (2, 10):
        loop = lp.Loop(10, ( k + 1 ) * 0.01, 8, -1, -1)
        energ = loop.Quantum_Monte_Carlo()
        fichier.write("The mean energy for a periodic chain of length 8 with Jx, Jz, beta = -1, -1, " + str((k+1) * 0.01) + " is " + 
                      str(np.mean(energ)) + '+/-' + str(np.std(energ)/np.sqrt(len(energ))) + "\n")
"""
"""
x = [2]
yerr = []
y = []
for k in x:
    loop = lp.Loop(k, 1. / k, 4, 1, 1)
    energ = loop.Quantum_Monte_Carlo(n_cycles = 100)
    y += [np.mean(energ)]
    yerr += [np.std(energ/np.sqrt(len(energ)))]

plt.errorbar(x, y, yerr=yerr)
"""


loop = lp.Loop(4, 0.5, 4, 5., 0.5)
print(loop.w, loop.w11, loop.w12, loop.w22, loop.w24, loop.w31, loop.w34)

energy = loop.Quantum_Monte_Carlo(n_cycles = 1000)
print('beta = ' + str(loop.m_trotter * loop.dtau) + '\nJx = ' + str(loop.Jx) +
      '\nJz = ' + str(loop.Jz) + '\nenergy = ' + str(np.mean(energy)))

fig = plt.figure(figsize = (10,15))
plt.imshow(loop.creategraph(), cmap = 'Greys_r')

#def make_frame(i):
#    loop.QMC_step()
#    im.set_array(loop.createimage())
#    return im,
#
#
#animation.FuncAnimation(fig, make_frame, interval = 10)
    
