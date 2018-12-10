# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:01:28 2018

@author: etien
"""

import loopclass as lp
import pickle as pk

J_x = - 1.
J_z = - 1.

beta = 1.
n_spins = 100
m_trotter = 20
dtau = m_trotter / beta

n_cycles = 10000
length_cycle = 10

loop = lp.Loop(m_trotter, dtau, n_spins, J_x, J_z)

energy, magnet = loop.Quantum_Monte_Carlo(n_warmup = 10, n_cycles = n_cycles, length_cycle = length_cycle)

with open('hundred_spins_energy.txt', 'wb') as fichier:
    pk.dump(energy, fichier)
with open('hundred_spins_magnet.txt', 'wb') as fichier:
    pk.dump(magnet, fichier)