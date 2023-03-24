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
lr difefrent ffor each
latent Z vec dimension
distribution type of Z
training data distribution
'''

batch_size = 256
datalist = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/normal1.npy")
N = 3 # three atoms
Z = 4 # latent vector length

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

class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.fc1 = nn.Linear(3, 64, bias=True) #specify 3 coord input
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 1, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = nn.Sigmoid()(x) #REMOVE for Wasserstien
        return x

def real (data, batch_size): #returns a batch of real data starting at a random starting point and the next "batch_size" elements
    index = np.random.randint(0,data.shape[0], batch_size)
    return data[index]
    
nn_G = Generator(Z) # Construct Networks using their generative classes
nn_D = Discrim()

criterion = nn.BCELoss() #maybe change to nn.CrossEntropyLoss()
D_op = optim.Adam(nn_D.parameters(), lr=1e-6) #RMSprop, maybe smaller lrs. 10^-3, 10^-4
G_op = optim.Adam(nn_G.parameters(), lr=1e-5)

D_label_r = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value. REAL identifiers as real
D_label_f = torch.full(size=(batch_size, 1), fill_value=0)  # Returns a tensor of size size filled with fill_value. FAKE identifiers as fake
G_label = torch.full(size=(batch_size, 1), fill_value=1) # Returns a tensor of size size filled with fill_value. FAKE identifiers as real

Cost = []
Cost_std = []

epochs = 900
updates = 20

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
        latent_vector = torch.empty(batch_size, Z).normal_(0, 1) # latent vector, zero centred unit variance Gaussian Distribution
        fake_data = nn_G(latent_vector)
        output = nn_D(fake_data.float())
        D_loss_f = criterion(output, D_label_f)
        D_loss_f.backward()
        D_op.step()

#       G - Train
            
       # try new latent
  
        G_op.zero_grad()
        fake_data = nn_G(latent_vector)
        output = nn_D(fake_data)
        G_loss = criterion(output, G_label)
        G_loss.backward()
        G_op.step()

#        Statistics

        cost_i.append([D_loss_r.item(), D_loss_f.item(), G_loss.item()])

    Cost.append(np.mean(cost_i, axis=0))
    Cost_std.append(np.std(cost_i, axis=0))
    print("Epoch: %i, D_Lr: %.4f, D_Lf: %.4f, G_L: %.4f" % (T, Cost[-1][0], Cost[-1][1], Cost[-1][2])) #first arg should be -1
    
    job.append(Cost[-1][1])
    on.append(Cost[-1][0])
    jambon.append(Cost[-1][2]) #again these 1s are -1
    tf.append(T)

latent_vector = torch.empty(10,000, Z).normal_(0, 1).float()
fake = nn_G(latent_vector).detach().numpy()

np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/gan1batch.npy",  fake) # save 10,000 fake molecules
torch.save(nn_G.state_dict(), '/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Trained GANs/gan3coord.h5')

plt.figure() #  Loss vs epoch
plt.scatter(tf, job, color = blu, marker = '.', label = ('D Loss fake'))
plt.scatter(tf, on, color = purp, marker  = '.', label = ('D Loss real'))
plt.scatter(tf, jambon, color = gren, marker = '.', label = ('G Loss'))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('D and G Loss')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/gan3loss.eps', bbox_inches='tight')

# Fake Molecules
plt.figure()
plt.title('Molecules')

b,c,d = [0],[1],[2]

for i in range (0,20):
    RESULT = fake_data.detach().numpy().tolist()
    samplerfaker = RESULT[random.randint(0,batch_size)]

    x2 = itemgetter(*b)(samplerfaker)
    x3 = itemgetter(*c)(samplerfaker)
    y3 = itemgetter(*d)(samplerfaker)
    
    plt.plot([0, x2, x3], [0, 0, y3], color = purp, marker = 'o',  label = ('Fake'))
    plt.plot([0, x3, x2], [0, y3, 0], color = purp, marker = 'o')
    
    i += 1

# Real Molecules

for i in range (0,20):
    samplereal = datalist[i+3]

    b = [0]
    c = [1]
    d = [2]

    x2 = itemgetter(*b)(samplereal)
    x3 = itemgetter(*c)(samplereal)
    y3 = itemgetter(*d)(samplereal)
    
    
    plt.plot([0, x2, x3], [0, 0, y3], color = blu, marker = 'o',  label = ('Real'))
    plt.plot([0, x3, x2], [0, y3, 0], color = blu, marker = 'o')
    
    i += 1
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/gan3.eps', bbox_inches='tight')
#plt.legend()
plt.show()


