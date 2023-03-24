#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

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

batch_size = 50	
datalist = np.load("datalist2.npy")
N = 3 # three atoms
Z = 5 # latent vector length

class Generator(nn.Module):
    def __init__(self, N, Z):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(Z, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, N*2, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # pass through hidden layers
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        """
        only for centering
        """
#        # reshape
#        x = x.view(-1, self.N_at, 3)
#        # centre molecules at origin
#        x = x - x.mean(1, keepdim=True)
        return x


class Discrim(nn.Module):
    def __init__(self, N):
        super(Discrim, self).__init__()
        self.fc1 = nn.Linear(N*2, 64, bias=True)
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

#def lat (size, Z):  #generates a z dimensional vector with size = batch_size. Don't actually need this as it's done in a line within the generator
#    latent = []
#    for i in range(0, size):
#        latent.append(np.random.randn(Z))
#    return np.asarray(latent)

def real (data, batch_size): #returns a batch of real data starting at a random starting point and the next "batch_size" elements
    index = np.random.randint(0,data.shape[0], batch_size)
    return data[index]
    
#latent_vector = torch.from_numpy(lat(batch_size, Z))
real_batch = torch.from_numpy(real(datalist, batch_size))

nn_G = Generator(N, Z)
nn_D = Discrim(N)


criterion = nn.BCELoss() #maybe change to nn.CrossEntropyLoss()
D_op = optim.Adam(nn_D.parameters(), lr=1e-3)
G_op = optim.Adam(nn_G.parameters(), lr=1e-3)

D_label_r = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value.
D_label_f = torch.full(size=(batch_size, 1), fill_value=0)  # Returns a tensor of size size filled with fill_value.
G_label = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value.

Cost = []
Cost_std = []

epochs = 250
updates = 20


for T in range(1, epochs+1):
    cost_i = []
    for i in range(updates):

        D_op.zero_grad()

        #Load a real batch of data

        output = nn_D(real_batch.float())
        D_loss_r = criterion(output, D_label_r)
        D_loss_r.backward()  # computes gradient

        seeds = torch.empty(batch_size, Z).uniform_(-1, 1)
        
        """
        D - Train on Fake Data
        """
        fake_data = nn_G(seeds)
        output = nn_D(fake_data.float())
        D_loss_f = criterion(output, D_label_f)
        D_loss_f.backward()

        D_op.step()


        """
        G - Train
        """
        G_op.zero_grad()
#        seeds = torch.empty(batch_size, Z).uniform_(-1, 1)
        fake_data = nn_G(seeds)
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

#    if T % checkpoint == 0:
#        print("Saving Model")
#        torch.save(nn_G.state_dict(), save_to + 'G%i' % T)

#when D is outputting half, then both losses to tend to log 2 ~ .693



"""
G_nn .load_state_dict(torch,load( "location" )

to load the thing then get latnt vec like:


GS['seeds']).uniform_(-1, 1)
fake_data = nn_G(seeds)
output = nn_D(fake_data)

to make a molecule
"""
plt.figure()
plt.style.use('ggplot')

plt.title("real blue, fake red")

for i in range (0,20):
    RESULT = fake_data.detach().numpy().tolist()
    samplerfaker = RESULT[random.randint(0,batch_size)]

    from operator import itemgetter
    c = [1, 3, 5]
    b = [0, 2, 4]
    print()

    xcoord_f = itemgetter(*b)(samplerfaker)
    ycoord_f = itemgetter(*c)(samplerfaker)
    
    plt.plot(xcoord_f, ycoord_f, 'ro')
    
#    print ("interatomic distances", np.sqrt((xcoord_f[0]-xcoord_f[1])**2+(ycoord_f[0]-ycoord_f[1])**2), np.sqrt((xcoord_f[0]-xcoord_f[2])**2+(ycoord_f[0]-ycoord_f[2])**2), np.sqrt((xcoord_f[2]-xcoord_f[1])**2 + (ycoord_f[2]-ycoord_f[1])**2))
    i += 1
    
for i in range (0,20):
    samplereal = datalist[random.randint(0,batch_size)]

    c = [1, 3, 5]
    b = [0, 2, 4]

    xcoord_r = itemgetter(*b)(samplereal)
    ycoord_r = itemgetter(*c)(samplereal)

    plt.plot(xcoord_r, ycoord_r, 'bo')

  
i += 1


plt.show()


#print ("interatomic distances", np.sqrt((xcoord[0]-xcoord[1])**2+(ycoord[0]-ycoord[1])**2), np.sqrt((xcoord[0]-xcoord[2])**2+(ycoord[0]-ycoord[2])**2), np.sqrt((xcoord[2]-xcoord[1])**2 + (ycoord[2]-ycoord[1])**2))
"""
NOTES

need to make sure  generator gives right distribution not just the right mean!!

multi target error, then squeeze the labels

"""
