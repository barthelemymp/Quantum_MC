# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 01:03:57 2018

@author: bmeyn
"""

import localclassnew as lc
import confignew as cf
import numpy as np


#
#c = cf.config(2,0.5,4,1,1)
#e = c.Quantum_Monte_Carlo(n_cycles=1000000, length_cycle=50)
#np.save("m2_dt0,5_ns4_jx1_jz1_lc10_mil",e)
##
#c = cf.config(3,1/3,4,1,1)
#e = c.Quantum_Monte_Carlo(n_cycles=1000000, length_cycle=100)
#np.save("m3_dt0,33_ns4_jx1_jz1_lc10_mil",e)
##
#c = cf.config(4,0.25,4,1,1)
#e = c.Quantum_Monte_Carlo(n_cycles=1000000, length_cycle=100)
#np.save("m4_dt0,25_ns4_jx1_jz1_lc10_mil",e)

#c = cf.config(5,0.2,4,1,1)
#e = c.Quantum_Monte_Carlo(n_cycles=1000000, length_cycle=100)
#np.save("m5_dt0,5_ns4_jx1_jz1_lc10_mil",e)

c = cf.config(6,1/6,4,1,1)
e = c.Quantum_Monte_Carlo(n_cycles=1000000, length_cycle=100)
np.save("m6_dt0,16_ns4_jx1_jz1_lc10_mil",e)