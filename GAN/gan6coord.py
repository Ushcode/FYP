#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from operator import itemgetter

blu = '#206b99'
gren = '#b4ecb4'
purp =  '#A16E83'
plum = purp


# Changeable Parameters
'''
batch
epochs
architecture
lr
latent Z vec dimension
distribution type of Z
training data distribution
'''

batch_size = 128
datalist = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/normal2.npy") # uniform or normal
N = 3 # three atoms
Z = 3 # latent vector length

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

def real (data, batch_size): #returns a batch of real data starting at a random starting point and the next "batch_size" elements
    index = np.random.randint(0,data.shape[0], batch_size)
    return data[index]
    
#latent_vector = torch.from_numpy(lat(batch_size, Z))
#real_batch = torch.from_numpy(real(datalist, batch_size))

nn_G = Generator(N, Z)
nn_D = Discrim(N)


criterion = nn.BCELoss() #maybe change to nn.CrossEntropyLoss() STARTED w nn.BCELoss,  there's also BCEWithLogitsLoss()
D_op = optim.Adam(nn_D.parameters(), lr=1e-6)
G_op = optim.Adam(nn_G.parameters(), lr=1e-5)

D_label_r = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value.
D_label_f = torch.full(size=(batch_size, 1), fill_value=0)  # Returns a tensor of size size filled with fill_value.
G_label = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value.

Cost = []
Cost_std = []

epochs = 1000
updates = 40

job, tf, on, jambon = [],[],[],[]

for T in range(1, epochs+1):
    cost_i = []
    for i in range(updates):

        D_op.zero_grad()

        #Load a real batch of data
        real_batch = torch.from_numpy(real(datalist, batch_size))
        output = nn_D(real_batch.float())
        D_loss_r = criterion(output, D_label_r)
        D_loss_r.backward()  # computes gradient

#       D - Train on Fake Data
        latent_vector = torch.empty(batch_size, Z).normal_(0, 1) # GAUSSIAN latent vec
        fake_data = nn_G(latent_vector)
        output = nn_D(fake_data.float())
        D_loss_f = criterion(output, D_label_f)
        D_loss_f.backward()
        D_op.step()

#       G - train

        G_op.zero_grad()
#        seeds = torch.empty(batch_size, Z).uniform_(-1, 1)
        fake_data = nn_G(latent_vector)
        output = nn_D(fake_data)
        G_loss = criterion(output, G_label)
        G_loss.backward()
        G_op.step()

#        Statistics
        cost_i.append([D_loss_r.item(), D_loss_f.item(), G_loss.item()])

    Cost.append(np.mean(cost_i, axis=0))
    Cost_std.append(np.std(cost_i, axis=0))
    print("Epoch: %i, D_Lr: %.4f, D_Lf: %.4f, G_L: %.4f" % (T, Cost[-1][0], Cost[-1][1], Cost[-1][2]))

    job.append(Cost[-1][1])
    on.append(Cost[-1][0])
    jambon.append(Cost[-1][2])
    tf.append(T)

#    if T % checkpoint == 0:
#        print("Saving Model")
#        torch.save(nn_G.state_dict(), save_to + 'G%i' % T

plt.figure() #  Loss vs epoch
plt.scatter(tf, job, color = blu, marker = '.', label = ('D Loss fake'))
plt.scatter(tf, on, color = plum, marker  = '.', label = ('D Loss real'))
plt.scatter(tf, jambon, color = gren, marker = '.', label = ('G Loss'))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('D and G Loss')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/gan6loss.eps', bbox_inches='tight')

latent_vector = torch.empty(10,000, Z).normal_(0, 1).float()
fake = nn_G(latent_vector).detach().numpy()

np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/gan2.npy",  fake) # save 10,000 fake molecules, same as real dataset size
torch.save(nn_G.state_dict(), '/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Trained GANs/gan6coord.h5') # save the trained generator

#plot 20 samples from generator
plt.figure()
plt.title("Real (blue), fake (purple)")

c,b = [1, 3, 5], [0, 2, 4]



for i in range (0,20):
    RESULT = fake_data.detach().numpy().tolist()
    samplefake = RESULT[random.randint(0,batch_size - 1)]
    
    xcoord_f = itemgetter(*b)(samplefake)
    ycoord_f = itemgetter(*c)(samplefake)
    
    plt.scatter(xcoord_f, ycoord_f, color = purp , marker = 'o')
    
    i += 1

#plot 20 samples from real batch

#plt.figure()

for i in range (0,20):
    samplereal = datalist[random.randint(0,batch_size)]
    
    xcoord_r = np.array(itemgetter(0,2,4)(samplereal))
    ycoord_r = np.array(itemgetter(1,3,5)(samplereal))

    plt.scatter(xcoord_r, ycoord_r, color = blu, marker = 'o')
    
    #print ("interatomic distances", np.sqrt((xcoord_f[0]-xcoord_f[1])**2+(ycoord_f[0]-ycoord_f[1])**2), np.sqrt((xcoord_f[0]-xcoord_f[2])**2+(ycoord_f[0]-ycoord_f[2])**2), np.sqrt((xcoord_f[2]-xcoord_f[1])**2 + (ycoord_f[2]-ycoord_f[1])**2))
    i += 1

plt.xlabel('x')
plt.ylabel('y')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/gan6.eps', bbox_inches='tight')
plt.show()

"""
NOTES
need to make sure  generator gives right distribution not just the right mean!!
multi target error, then squeeze the labels

too many  epochs gives bad  results, mode  collapse  or  complete  failure
"""


