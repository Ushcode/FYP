#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from operator import itemgetter

"""
1) Load Ground State Data

2) Make distortions (Small)

3) Edit out/inpu of G&D

4) Define batch size (e.g. 32/64)

5) Standardizing data
"""

# Changeable Parameters
'''
Z
batch
epochs
architecture
lr
latent Z vec dimension
'''

batch_size = 128
datalist = np.load("datalist1.npy")
N = 3 # three atoms
Z = 3 # latent vector length

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
        """
        only for centering
        """
        return x


class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.fc1 = nn.Linear(3, 64, bias=True) #specify 3 coord input
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 1, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # pass through hidden layers
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        return x

def real (data, batch_size): #returns a batch of real data starting at a random starting point and the next "batch_size" elements
    index = np.random.randint(0,data.shape[0], batch_size)
    return data[index]
    
latent_vector = torch.empty(batch_size, Z).uniform_(-1, 1)
real_batch = torch.from_numpy(real(datalist, batch_size))

nn_G = Generator(N)
nn_D = Discrim()


criterion = nn.BCELoss() #maybe change to nn.CrossEntropyLoss()
D_op = optim.Adam(nn_D.parameters(), lr=1e-3)
G_op = optim.Adam(nn_G.parameters(), lr=1e-3)

D_label_r = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value.
D_label_f = torch.full(size=(batch_size, 1), fill_value=0)  # Returns a tensor of size size filled with fill_value.
G_label = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value.

Cost = []
Cost_std = []

epochs = 300
updates = 20


for T in range(1, epochs+1):
    cost_i = []
    for i in range(updates):

        D_op.zero_grad()

        #Load a real batch of data

        output = nn_D(real_batch.float())
        D_loss_r = criterion(output, D_label_r)
        D_loss_r.backward()  # computes gradient

        
        """
        D - Train on Fake Data
        """
        fake_data = nn_G(latent_vector)
        output = nn_D(fake_data.float())
        D_loss_f = criterion(output, D_label_f)
        D_loss_f.backward()

        D_op.step()


        """
        G - Train
        """
        G_op.zero_grad()
        fake_data = nn_G(latent_vector)
        output = nn_D(fake_data)
        G_loss = criterion(output, G_label)
        G_loss.backward()
        G_op.step()


        """
        Statistics
        """
        cost_i.append([D_loss_r.item(), D_loss_f.item(), G_loss.item()])

    Cost.append(np.mean(cost_i, axis=0))
    Cost_std.append(np.std(cost_i, axis=0))
    print("Epoch: %i, D_Lr: %.4f, D_Lf: %.4f, G_L: %.4f" % (T, Cost[-1][0], Cost[-1][1], Cost[-1][2]))

b = [0]
c = [1]
d = [2]
    
# Fake Molecules
plt.figure()
plt.style.use('ggplot')
plt.title("Fake Molecules")

#for i in range (0,20):
#    RESULT = fake_data.detach().numpy().tolist()
#    samplerfaker = RESULT[random.randint(0,batch_size)]
#
#    x2 = itemgetter(*b)(samplerfaker)
#    x3 = itemgetter(*c)(samplerfaker)
#    y3 = itemgetter(*d)(samplerfaker)
#    
#    plt.style.use('ggplot')
#    plt.plot([0, x2, x3], [0, 0, y3], 'ro-')
#    plt.plot([0, x3, x2], [0, y3, 0], 'ro-')
#    
#    i += 1

## Real Molecules
#plt.figure()
#plt.title("Real Molecules")
#
#for i in range (0,20):
#    samplereal = datalist[i+3]
#
#    b = [0]
#    c = [1]
#    d = [2]
#
#    x2 = itemgetter(*b)(samplerfaker)
#    x3 = itemgetter(*c)(samplerfaker)
#    y3 = itemgetter(*d)(samplerfaker)
#
#    plt.style.use('ggplot')
#    plt.plot([0, x2, x3], [0, 0, y3], 'ro-')
#    plt.plot([0, x3, x2], [0, y3, 0], 'ro-')
#
#    i += 1
#
#plt.show()
