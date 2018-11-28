import localclass as st
import loopclass as lp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8)
s = st.States(4,1,8,1,1)
for k in range(1000):
    if k%100 == 0:
        s.splitline()
    s.local_update()
    #s.createimage()



loop = lp.Loop(4,1,8,1,1)
loop.pattern = s.pattern

#loop.createimage()
#loop.creategraph()

#print(loop.pattern)

loop.spins = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0.]])
loop.spins_to_pattern()
loop.set_total_graph()
#loop.createimage()
loop.creategraph()

res = ((0,1), (0,0))
c = (7,1)
x = [0]
y = [1]
while res[0] != c:
    res = loop.loop_from_graph(res[0], res[1])
    x += [res[0][0]]
    y += [res[0][1]]
    print(res)

fig = plt.figure(figsize= (10,10))
plt.scatter(y, x)
