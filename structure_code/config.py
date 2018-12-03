he# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:03:23 2018

@author: bmeyn
"""

import numpy as np
import numpy.random as rnd
#import cv2
import matplotlib.pyplot as plt
import localclass as lc


class config:
    """A Feynman path in position/imaginary time space"""

    def __init__(self, m_trotter, dtau, n_spins, Jx, Jz, mode = 'local', ):
        self.m_trotter = m_trotter #division of the temporary time"
        self.dtau = dtau
        self.n_spins = n_spins
        self.Jx = Jx
        self.Jz = Jz
        self.mode = mode


    def compute_energy_autocorrelation(self,n_splitline,n_localupdate):
        if(self.mode is 'local'):
            s = lc.States(self.m_trotter,self.dtau,self.n_spins,self.Jx,self.Jz)
            for i in range (100): # warm
                s.basic_move_simple(n_splitline,n_localupdate)
            E = s.total_energy()
            En_list = [E]
            for k in range (15): # compute energie evolution
                dE, _ = s.basic_move(n_splitline, n_localupdate)
                E +=dE
                En_list += [E]
            Energies = np.array(En_list[:10])
            sqmean = np.mean(Energies * Energies)
            meansq = np.mean(Energies)**2
            autocorr = np.zeros(5)
            for t in range (5):
                Energies_tmove = (En_list[t:t+10])
                corr = np.mean(Energies * Energies_tmove)
                autocorr[t] = (corr - meansq)/(sqmean - meansq)
            x=np.linspace(0,4,5)
            plt.plot(x,autocorr)
            print("enelist",En_list)
            print("Energies",Energies)
            print("sqmean",sqmean)
            print("meansq",meansq)
            return
            
            
        if(self.mode is 'loop'):
            return
        
        
    def Quantum_Monte_Carlo(self,n_warmup=100,n_cycles = 200,length_cycle = 100):
        state = lc.States(self.m_trotter, self.dtau, self.n_spins,self.Jx, self.Jz)
        state.basic_move_simple(10,30)
        
        energ = np.zeros(n_cycles)
        # Monte Carlo simulation
        for n in range(n_warmup+n_cycles):
            print(n)
            # Monte Carlo moves
            for l in range(length_cycle):
                dE, dw = state.stoch_move(0.5)

                
                #self.autremodif
            # measures
            if n >= n_warmup:
                e = state.total_energy()
                #state.createimage()
                energ[n-n_warmup] = e
                #print("ener",e)
        print('Energy:', np.mean(energ), '+/-', np.std(energ)/np.sqrt(len(energ)))
        return energ


def QMC_mean_chain(m_trotter, dtau, n_spins, Jx, Jz):
    with open('QMCMeanEnergyJx' + str((n_spins, Jz, m_trotter, dtau)) + '.txt', 'w') as fichier:
        for Jx in range(0, 5):
            conf = config(m_trotter, dtau, n_spins, Jx, Jz)
            energy = conf.Quantum_Monte_Carlo(1000)
            beta = dtau* m_trotter
            fichier.write("The QMC mean energy for a periodic chain of length " 
                          + str(n_spins) + " with Jx,Jz, beta = " + str((-Jx,Jz, beta)) + " is " + 
                          str(energy) + '\n')
    