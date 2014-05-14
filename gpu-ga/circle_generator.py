import math

m = 10

print m
for i in range(m):
    print str(math.cos(i / float(m) * math.pi * 2.0)) + " " + str(math.sin(i / float(m) * math.pi * 2.0))