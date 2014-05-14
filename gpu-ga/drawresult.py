import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys

map_file = sys.argv[1] # 'qa194.in'
result_file = sys.argv[2]

f = open(result_file, 'r')
distance = float(f.readline())
path = f.readline().strip().split(' ')
f.close()

f = open(map_file, 'r')
count = int(f.readline())

cities = []
verts = []
codes = []

for line in f:
	params = line.strip().split(' ')
	cities.append((float(params[0]), float(params[1])))
	codes.append(Path.LINETO)

for city in path:
	verts.append(cities[int(city)])

f.close()

codes[0] = Path.MOVETO
verts.append((0., 0.))
codes.append(Path.CLOSEPOLY)

path2 = Path(verts, codes)

fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path2, facecolor='white', lw=2)
ax.add_patch(patch)
ax.set_xlim(0,5)
ax.set_ylim(0,5)
plt.show()