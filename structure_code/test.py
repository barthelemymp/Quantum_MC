import config as cf
import localclass as st
import numpy as np

"""
s = st.States(2, 0.025, 4, 0.0000001, -1)
s.createimage()
for k in range (5):
    print(s.splitline())
    s.createimage()
    print(s.local_update())
    s.createimage()
"""

conf = cf.config(10, 0.15, 8, -2, -1)
sta = st.States(1, 1, 4, -1, 0)
#s.splitline()
#s.splitline()
#s.createimage()

#conf.Quantum_Monte_Carlo(n_cycles = 1000)
"""
for k in range(100):
    if k % 10 == 0:
        print(s.splitline())
    print(s.local_update())
#    s.createimage()
#"""
    
#s.createimage()

"""
for k in range(10000):
    if k % 50 == 0:
        s.splitline()
    else:
        s.local_update()
    s.createimage()

    
print('end')
"""
