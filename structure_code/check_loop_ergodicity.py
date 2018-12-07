# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:45:47 2018

@author: etien
"""

import loopclass as lp
import numpy as np

np.random.seed(12987848)

"""
This scripts allows the check of the ergodicity for given values of m and 4 spins

theoretical values are with periodic boundaries conditions
m = 2: 90
m = 3: 546
m = 4: 3618
m = 5: 25218
m = 6: 181122
"""

def check_ergodicity(m, Jx, Jz, steps):
    loop = lp.Loop(m, 1., 4, Jx, Jz)
    loop.w11 = 0.5
    loop.w12 = 0.5
    loop.w22 = 0.5
    loop.w24 = 0.5
    loop.w31 = 0.5
    loop.w34 = 0.5
    
    pattern_done = {}
    for k in range(steps):
        print(k)
        loop.QMC_step()
        pattern_done[loop.pattern_to_string()] = 1
    
    print('with m = ' + str(m) + ' there are ' + str(len(pattern_done)) + ' configurations')
    return len(pattern_done)

check_ergodicity(4, 1, 0.5, 1000000)