 #!/usr/bin/python

"""
Now Changing coordinate system to include all 6 dof (x_i^j) Same potential etc
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random

plum =  '#A16E83'
#3 particles => 6 DOF. Keep particle 1 at origin and particle 2 on the x axis. Now include zeros for particle 1 and 2

x1var, y1var = random.uniform(-0.1, 0.1), random.uniform(-0.5, 0.5)
x2var, y2var = random.uniform(1.0, 2.0), random.uniform(1.0, 2.0)
x3var, y3var = random.uniform(2.0, 3.0), random.uniform(1.5, 2.5)

# Bond energy from LJ potential. e is potential well depth; R0 is desired mimimum of potential

R0 = 1.0
e = 10

#Determine forces on particles as the Gradient of the potential

def F1x(x1,y1,x2,y2,x3,y3):
    return e*((12*R0**12 * (x3-x1)*((x3-x1)**2+(y3-y1)**2)**-7) + (-6*R0**6 * (x3-x1)*((x3-x1)**2+(y3-y1)**2)**-4) + (12*R0**12 * (x2-x1)*((x2-x1)**2+(y2-y1)**2)**-7) + (-6*R0**6 * (x2-x1)*((x2-x1)**2+(y2-y1)**2)**-4))
    
def F1y(x1,y1,x2,y2,x3,y3):
    return e*((12*R0**12 * (y3-y1)*((x3-x1)**2+(y3-y1)**2)**-7) + (-6*R0**6 * (y3-y1)*((x3-x1)**2+(y3-y1)**2)**-4) + (12*R0**12 * (y2-y1)*((x2-x1)**2+(y2-y1)**2)**-7)  + (-6*R0**6 * (y2-y1)*((x2-x1)**2+(y2-y1)**2)**-4))

def F2x(x1,y1,x2,y2,x3,y3):
    return e*((-12*R0**12 * (x2-x1)*((x2-x1)**2+(y2-y1)**2)**-7) + (6*R0**6 * (x2-x1)*((x2-x1)**2+(y2-y1)**2)**-4) + (12*R0**12 * (x3-x2)*((x3-x2)**2+(y3-y2)**2)**-7) + (-6*R0**6 * (x3-x2)*((x3-x2)**2+(y3-y2)**2)**-4))

def F2y(x1,y1,x2,y2,x3,y3):
    return e*((-12*R0**12 * (y2-y1)*((x2-x1)**2+(y2-y1)**2)**-7) + (6*R0**6 * (y2-y1)*((x2-x1)**2+(y2-y1)**2)**-4) + (12*R0**12 * (y3-y2)*((x3-x2)**2+(y3-y2)**2)**-7)  + (-6*R0**6 * (y3-y2)*((x3-x2)**2+(y3-y2)**2)**-4))
    
def F3x(x1,y1,x2,y2,x3,y3):
    return e*((-12*R0**12 * (x3-x1)*((x3-x1)**2+(y3-y1)**2)**-7) + (6*R0**6 * (x3-x1)*((x3-x1)**2+(y3-y1)**2)**-4) + (-12*R0**12 * (x3-x2)*((x3-x2)**2+(y3-y2)**2)**-7) + (6*R0**6 * (x3-x2)*((x3-x2)**2+(y3-y2)**2)**-4))

def F3y(x1,y1,x2,y2,x3,y3):
    return e*((-12*R0**12 * (y3-y1)*((x3-x1)**2+(y3-y1)**2)**-7) + (6*R0**6 * (y3-y1)*((x3-x1)**2+(y3-y1)**2)**-4) + (-12*R0**12 * (y3-y2)*((x3-x2)**2+(y3-y2)**2)**-7)  + (6*R0**6 * (y3-y2)*((x3-x2)**2+(y3-y2)**2)**-4))
    
d = 0.01
error = 0.000001
def F(F1x, F1y, F2x, F2y, F3x, F3y):
    return np.abs(F1x) + np.abs(F1y) + np.abs(F2x) + np.abs(F2y) + np.abs(F3x) + np.abs(F3y)

iterations = 0

while F(F1x(x1var,y1var,x2var,y2var,x3var,y3var), F1y(x1var,y1var,x2var,y2var,x3var,y3var), F2x(x1var,y1var,x2var,y2var,x3var,y3var), F2y(x1var,y1var,x2var,y2var,x3var,y3var), F3x(x1var,y1var,x2var,y2var,x3var,y3var), F3y(x1var,y1var,x2var,y2var,x3var,y3var)) > error :
    
    x1var += -F1x(x1var,y1var,x2var,y2var,x3var,y3var) * d
    
    y1var += -F1y(x1var,y1var,x2var,y2var,x3var,y3var) * d
    
    x2var += -F2x(x1var,y1var,x2var,y2var,x3var,y3var) * d
    
    y2var += -F2y(x1var,y1var,x2var,y2var,x3var,y3var) * d
    
    x3var += -F3x(x1var,y1var,x2var,y2var,x3var,y3var) * d

    y3var += -F3y(x1var,y1var,x2var,y2var,x3var,y3var) * d

    iterations +=1

    print ('P1', x1var, y1var, 'P2', x2var, y2var, 'P3', x3var, y3var)
    
R12 = math.sqrt((x2var-x1var)**2+(y2var-y1var)**2)
R13 = math.sqrt((x3var-x1var)**2+(y3var-y1var)**2)
R23 = math.sqrt((x3var-x2var)**2+(y3var-y2var)**2)


print('%d iterations \n \n' %(iterations))
print('Interatomic Distances R_ij', R12, R13, R23)

plt.figure()
plt.style.use('ggplot')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([x1var, x2var, x3var], [y1var, y2var, y3var], color = plum, marker = 'o')
plt.plot([x1var, x3var, x2var], [y1var, y3var, y2var], color = plum, marker = 'o')
#plt.title("Arrangement of three Particles")
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/triangle2.eps', bbox_inches='tight')
plt.show()


# Keeping track of energies

E12 = e*((R0/R12)**12 - (R0/R12)**6)
E13 = e*((R0/R13)**12 - (R0/R13)**6)
E23 = e*((R0/R23)**12 - (R0/R23)**6)

E = E12 + E13 + E23

print ("Minimum Energy E =", E)

####################################################################
# Generate Neural network datasets #################################
####################################################################
x1i, y1y, x2i, y2i, x3i, y3i = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

uniform, normal = [],[]
nrglist_u, nrglist_n = [],[]

for i in range (10000):
    x1i = x1var + random.uniform(-0.1, 0.1) # generate a uniform distribution
    y1i = y1var + random.uniform(-0.1, 0.1)
    x2i = x2var + random.uniform(-0.1, 0.1)
    y2i = y2var + random.uniform(-0.1, 0.1)
    x3i = x3var + random.uniform(-0.1, 0.1)
    y3i = y3var + random.uniform(-0.1, 0.1)
    
    x1in = np.random.normal(x1var, 0.05) # generate a Gaussian distribution
    y1in = np.random.normal(y1var, 0.05)
    x2in = np.random.normal(x2var, 0.05)
    y2in = np.random.normal(y2var, 0.05)
    x3in = np.random.normal(x3var, 0.05)
    y3in = np.random.normal(y3var, 0.05)

    uniform.append([x1i, y1i, x2i, y2i, x3i, y3i])
    normal.append([x1in, y1in, x2in, y2in, x3in, y3in])

    R12i = math.sqrt((x2i-x1i)**2+(y2i-y1i)**2) # Interatomic distances
    R13i = math.sqrt((x3i-x1i)**2+(y3i-y1i)**2)
    R23i = math.sqrt((x3i-x2i)**2+(y3i-y2i)**2)
    
    E12i = e*((R0/R12i)**12 - (R0/R12i)**6) # Energy from LJ
    E13i = e*((R0/R13i)**12 - (R0/R13i)**6)
    E23i = e*((R0/R23i)**12 - (R0/R23i)**6)
    
    R12in = math.sqrt((x2in-x1in)**2+(y2in-y1in)**2) # Interatomic distances
    R13in = math.sqrt((x3in-x1in)**2+(y3in-y1in)**2)
    R23in = math.sqrt((x3in-x2in)**2+(y3in-y2in)**2)
    
    E12in = e*((R0/R12in)**12 - (R0/R12in)**6) # Energy from LJ
    E13in = e*((R0/R13in)**12 - (R0/R13in)**6)
    E23in = e*((R0/R23in)**12 - (R0/R23in)**6)


    Ei = E12i + E13i + E23i
    nrglist_u.append(Ei)
    
    Ein = E12in + E13in + E23in
    nrglist_n.append(Ein)

np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/uniform2", uniform)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/normal2", normal)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/nrg_u2", nrglist_u)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/nrg_n2", nrglist_n)
