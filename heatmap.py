 #!/usr/local/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import torch.nn as nn
import torch
from matplotlib.colors import LinearSegmentedColormap
import random
from operator import itemgetter

e=10
R0=1
Z=4

latent_vector = torch.empty(10000, Z).uniform_(0, 1).float()

############# GENERATORS #############
class Generator(nn.Module):
    def __init__(self, Z):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(Z, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 3, bias=True) #specify three coord output, indep of N this time
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # pass through hidden layers
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

Vanilla = Generator(Z)
Vanilla.load_state_dict(torch.load('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Trained GANs/vanilla.h5'))
Wasser = Generator(Z)
Wasser.load_state_dict(torch.load('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Trained GANs/Wasser.h5'))

outvan = Vanilla(latent_vector).detach().numpy()
outwas = Wasser(latent_vector).detach().numpy()

np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/outvan.npy",  outvan)
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/outwas.npy",  outwas)

Ev, Ew = np.zeros(len(latent_vector.numpy())),np.zeros(len(latent_vector.numpy()))

for i in range (len(outvan)):
#    x2van.append(outvan[i][0])
#    x3van.append(outvan[i][1])
#    y3van.append(outvan[i][2])
#
#    x2was.append(outwas[i][0])
#    x3was.append(outwas[i][1])
#    y3was.append(outwas[i][2])

    R12v = outvan[i][0]
    R13v = math.sqrt(outvan[i][1]**2 + outvan[i][2]**2)
    R23v = math.sqrt((outvan[i][1]-outvan[i][0])**2 + outvan[i][2]**2)

    R12w = outwas[i][0]
    R13w = math.sqrt(outwas[i][1]**2 + outwas[i][2]**2)
    R23w = math.sqrt((outwas[i][1]-outwas[i][0])**2 + outwas[i][2]**2)

    Ev[i] = e*((R0/R12v)**12 - (R0/R12v)**6) + e*((R0/R13v)**12 - (R0/R13v)**6) + e*((R0/R23v)**12 - (R0/R23v)**6)

    Ew[i] = e*((R0/R12w)**12 - (R0/R12w)**6) + e*((R0/R13w)**12 - (R0/R13w)**6) + e*((R0/R23w)**12 - (R0/R23w)**6)

v = np.reshape(Ev, (100,100), order='C')
w = np.reshape(Ew, (100,100), order='C')

#now have latent_vector, Ev, Ew

"""
    To make it
    - Use a seed dimension of 2
    - uniform distribution
    - make a 2d grid of inputs
    - for each input create a molecule and work out it's energy
"""

colors = ['#206B99', '#A16E83']  # blu to purp
n_bin = 1000  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)


plt.figure()
plt.title('Vanilla GAN')
plt.imshow(v, cmap=cm ) #, interpolation='spline16')
plt.colorbar()

plt.figure()
plt.title('Wasserstein GAN')
plt.imshow(w,cmap=cm ) #, interpolation='spline16')
plt.colorbar()

plt.show()  
