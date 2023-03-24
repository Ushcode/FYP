#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Model 0 database

import numpy as np
import random


ep = 10
r_min = 1.0

def LennardJones(r):
    return ep*((r_min/r)**12 - 2*(r_min/r)**6)

n= 4 #4 atom system 

#copy over your min x y arrays
    
xmin = [1.12023095,0.55553705,1.675768]
ymin = [0.97277745,0.97277745]
#combine into new package
c = [1.12023095, 0.55553705, 0.97277745, 1.675768, 0.97277745]
    
cmin =np.array(c)
    
a = 1.1202309525987397 #minimum side length
minenergy = -38.05065688997051

#edit original config by delta a lengths
#create a dataframe off x y energy

clist = [] #list of co-ordinate arrays
energy = [] #list of energies

clist.append(c)

energy.append(minenergy)


for i in range(0,10000): #generate 10 first
    co = np.array((0.0, 0.0, 1.12023095, 0.0, 0.55553705, 0.97277745, 1.675768, 0.97277745))
    c1 = [1.12023095, 0.55553705, 0.97277745, 1.675768, 0.97277745]
    for e in range(2,2*n):#  not going to change x[0] and y[0] - i now think i should
        if e!= 3:
            co[e] += (random.choice((-1,1)))*a*(random.random()/50)
    c2 = co.tolist()
    c3 = [g for g in c2 if g != 0.0]
    clist.append(c3)
    #calculate energy for each config 
    etot = 0
    for o in range(0,2*n,2): #for each atom
        h = np.array((co[o],co[o+1]))
        for u in range(0,2*n,2):
            if u!=o:
                b1 = np.array((co[u],co[u+1]))
                w = np.linalg.norm(h-b1) #calculate the distance between them
                etot += LennardJones(w)
    etot = etot/2   
    energy.append(etot)

      
#print(clist) 

#print(energy)

x = np.asarray(clist, dtype = 'double')
y = np.asarray(energy, dtype = 'double')


np.save('m0_x', x)
np.save('m0_y', y)
