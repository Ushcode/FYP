 #!/usr/bin/python

import numpy as np
import math
import matplotlib.pyplot as plt
import random

#3 particles => 6 DOF. Instead of using a cartesian coord system, keep track of only the distances between particles. Akin to a centre of mass frame. Lennard Jones simplifies, now we can have translational and rotational invariance

R12 = random.randint(1,10)
R23 = random.randint(1,10)
R13 = random.randint(1,10)

# Bond energy from LJ potential. e is potential well depth; R0 is desired mimimum of potential

R0 = 1.0
e = 10

E12 = e*((R0/R12)**12 - (R0/R12)**6)
E13 = e*((R0/R13)**12 - (R0/R13)**6)
E23 = e*((R0/R23)**12 - (R0/R23)**6)

E = E12 + E13 + E23

# Generate Neural network datasets

x1i, y1y, x2i, y2i, x3i, y3i = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

datalist = []
nrglist = []

for i in range (10000):
    R12i = R12 + random.uniform(-0.1, 0.1)
    R23i = R23 + random.uniform(-0.1, 0.1)
    R13i = R13 + random.uniform(-0.1, 0.1)

    datalist.append([R12i, R23i, R13i])

    
    E12i = e*((R0/R12i)**12 - (R0/R12i)**6)
    E13i = e*((R0/R13i)**12 - (R0/R13i)**6)
    E23i = e*((R0/R23i)**12 - (R0/R23i)**6)

    Ei = E12i + E13i + E23i
    
    nrglist.append(Ei)

np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/datalist3", datalist)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/nrglist3", nrglist)
