import localclass as st
import numpy as np

s = st.States(4,1,16,1,1)
s.splitline()
s.splitline()
s.createimage()


"""
for k in range(100):
    if k % 10 == 0:
        print(s.splitline())
    print(s.local_update())
#    s.createimage()
#"""

"""
for k in range(10000):
    if k % 50 == 0:
        s.splitline()
    else:
        s.local_update()
    s.createimage()

    
print('end')
"""
