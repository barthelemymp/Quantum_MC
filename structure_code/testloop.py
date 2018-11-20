import localclass as st
import loopclass as lp
import numpy as np

s = st.States(4,1,8,1,1)
for k in range(1000):
    if k%100 == 0:
        s.splitline()
    s.local_update()
    s.createimage()



loop = lp.Loop(4,1,8,1,1)
loop.pattern = s.pattern

loop.creategraph()
