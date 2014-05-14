import numpy as np
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

from numpy import genfromtxt

data = genfromtxt('benchmark.csv', delimiter=',')

x = data[:, 0]
y = data[:, 1]
z1 = data[:, 2]
z2 = data[:, 3]

ax = p3.Axes3D(p.figure())
s1 = ax.scatter(np.log2(x),y,z2, 'o', color="red", label='Serial version')
s2 = ax.scatter(np.log2(x),y,z1, 'o', color="blue", label='CUDA version')

ax.set_xlabel('Number of genes (2^n)')
ax.set_ylabel('Steps')
ax.set_zlabel('Run time')

p.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

p.show()