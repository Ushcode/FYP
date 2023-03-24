 #!/usr/bin/python 

"""
objective of this script is to find the optimal planar configuration of three particles,
each subject to a Lennard-Jones potential.

Then generates a dataset of normally distributed and of uniformly distributed points
"""

import numpy as np 
import math
import matplotlib.pyplot as plt

plum =  '#A16E83'
#############################################################################
# 3 particles => 6 DOF. Keep particle 1 at origin and particle 2 on the x axis
#############################################################################

x2var = 1.5
x3var = 2.5
y3var = 2.0

# Bond energy from LJ potential. e is potential well depth; R0 is desired mimimum of potential

R0 = 1.0
e = 10

# Determine forces on particles as the Gradient of the potential

def F2x(x2,x3,y3):
    return e*((-12*R0**12 *x2**-13) + (6*R0**6 *x2**-7) + (12*R0**12 *(x3-x2)*((x3-x2)**2+y3**2)**-7) + (-6*(x3-x2)*R0**6*((x3-x2)**2+y3**2)**-4))

def F3x(x2,x3,y3):
    return e*((-12*R0**12*(x3)*(x3**2+y3**2)**-7) + (6*x3*R0**6*(x3**2+y3**2)**-4) +  (-12*R0**12*(x3-x2)*((x3-x2)**2+y3**2)**-7) + (6*R0**6*(x3-x2)*((x3-x2)**2+y3**2)**-4))

def F3y(x2,x3,y3):
    return e*((-12*R0**12*y3*(x3**2+y3**2)**-7) + (6*y3*R0**6*(x3**2+y3**2)**-4) + (-12*R0**12*y3*((x3-x2)**2+y3**2)**-7) + (6*R0**6*y3*((x3-x2)**2+y3**2)**-4))

d = 0.01
error = 0.00000001
def F(F2x, F3x, F3y):
    return np.abs(F2x) + np.abs(F3x) + np.abs(F3y)

iterations = 0

while F(F2x(x2var, x3var, y3var), F3x(x2var, x3var, y3var), F3y(x2var, x3var, y3var)) > error :
    x2var += -F2x(x2var, x3var, y3var) * d
    x3var += -F3x(x2var, x3var, y3var) * d
    y3var += -F3y(x2var, x3var, y3var) * d
    iterations +=1
    
print('%d iterations' %(iterations))

plt.figure()
plt.style.use('ggplot')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([0, x2var, x3var], [0, 0, y3var], color = plum, marker = 'o')
plt.plot([0, x3var, x2var], [0, y3var, 0], color = plum, marker = 'o')
#plt.title("Arrangement of three Particles")

plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/triangle1.eps', bbox_inches='tight')

#R_ij is distance from particle i to particle j, three component vector.

R12 = x2var
R13 = math.sqrt(x3var**2 + y3var**2)
R23 = math.sqrt((x3var-x2var)**2 + y3var**2)

#Keeping track of energies

E12 = e*((R0/R12)**12 - (R0/R12)**6)
E13 = e*((R0/R13)**12 - (R0/R13)**6)
E23 = e*((R0/R23)**12 - (R0/R23)**6)

E = E12 + E13 + E23
 
#x_i = [x2var, x3var, y3var] - Vary this with random \delta added to individual components and
print ("Minimum Energy E =", E)

###################
# Generate DATA
###################


x2i, x3i, y3i = 0.0, 0.0, 0.0

import random

normal = []
uniform = []
nrg_uniform = []
nrg_normal = []


#un = []
#uny = []
#deux = []
#deuxy = []
#trois = []
#troisy =  []

for i in range (100000):
    x2i = x2var + random.uniform(-0.15, 0.15) # generate a uniform distribution
    x3i = x3var + random.uniform(-0.15, 0.15)
    y3i = y3var + random.uniform(-0.15, 0.15)
    
    x2in = np.random.normal(x2var, 0.05) # generate a Gaussian distribution
    x3in = np.random.normal(x3var, 0.05)
    y3in = np.random.normal(y3var, 0.05)
    
#    un.append(0)
#    uny.append(0)
#    deux.append(0)
#    deuxy.append(x2in)
#    trois.append(x3in)
#    troisy.append(y3in)

    normal.append([x2in, x3in, y3in])
    uniform.append([x2i, x3i, y3i])
    
#    f.write("x2_%d is %f \r\n" % (i, (x2i)) )
#    f.write("x3_%d is %f \r\n" % (i, (x3i)) )
#    f.write("y3_%d is %f \r\n" % (i, (y3i)) )

    R12i = x2i
    R13i = math.sqrt(x3i**2 + y3i**2)
    R23i = math.sqrt((x3i-x2i)**2 + y3i**2)
    
    E12i = e*((R0/R12i)**12 - (R0/R12i)**6)
    E13i = e*((R0/R13i)**12 - (R0/R13i)**6)
    E23i = e*((R0/R23i)**12 - (R0/R23i)**6)

    Ei = E12i + E13i + E23i
    
    R12in = x2in
    R13in = math.sqrt(x3in**2 + y3in**2)
    R23in = math.sqrt((x3in-x2in)**2 + y3in**2)
     
    E12in = e*((R0/R12in)**12 - (R0/R12in)**6)
    E13in = e*((R0/R13in)**12 - (R0/R13in)**6)
    E23in = e*((R0/R23in)**12 - (R0/R23in)**6)

    Ein = E12in + E13in + E23in
     
    nrg_uniform.append(Ei)
    nrg_normal.append(Ein)
    
#plt.figure()
#plt.title('randomly generated positions')
#plt.plot(un,uny, 'ro')
#plt.plot(deux, deuxy, 'bo')
#plt.plot(trois, troisy, 'go')

#plt.figure()
#plt.title('randomly generated normal energies')
#plt.plot(nrg_normal)
#plt.show()

uniform = np.array(uniform)
nrg_uniform = np.array(nrg_uniform)

normal = np.array(normal)
nrg_normal = np.array(nrg_normal)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/uniform1", uniform)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/nrg_uniform1", nrg_uniform)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/normal1", normal)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/nrg_normal1", nrg_normal)

