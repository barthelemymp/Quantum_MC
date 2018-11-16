import statesclass_test as st
import numpy as np

s = st.States(10,1,20,1,1)
s.splitline()
s.splitline()
s.createimage()

"""
for k in range(10000):
    if k % 500 == 0:
        s.splitline()
    else:
        s.local_update()

    
print('end')
"""
