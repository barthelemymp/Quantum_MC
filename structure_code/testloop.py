import localclass as st
import loopclass as lp
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(4)
#s = st.States(4,1,8,1,1)
#for k in range(1000):
#    if k%100 == 0:
#        s.splitline()
#    s.local_update()
    #s.createimage()



loop = lp.Loop(8,1,8,1,1)
#loop.pattern = s.pattern

#loop.createimage()
#loop.creategraph()

#print(loop.pattern)

#loop.spins = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
#                    [0., 1., 0., 0., 0., 0., 0., 0.],
#                    [0., 0., 1., 0., 0., 0., 0., 0.],
#                    [0., 0., 0., 1., 0., 0., 0., 0.],
#                    [0., 0., 0., 1., 0., 0., 0., 0.],
#                    [0., 0., 1., 0., 0., 0., 0., 0.],
#                    [0., 1., 0., 0., 0., 0., 0., 0.],
#                    [0., 1., 0., 0., 0., 0., 0., 0.]])
for k in range(10):
    loop.spins_to_pattern()
    loop.set_total_graph()
    loop.find_loops()
    loop.createimage()


#res = ((0,0), (0,0))
#c = (7,0)
#x = [0]
#y = [1]
#while res[0] != c:
#    res = loop.find_next(res[0], res[1])
#    x += [res[0][0]]
#    y += [res[0][1]]
#    k += 1
#    print(res)

#plt.subplot(111)
#plt.scatter(y, x)
#plt.xlim(0, 8)

