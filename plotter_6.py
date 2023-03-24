    #!/usr/loreal/bin/python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from operator import itemgetter
plt.style.use('ggplot')
import random
import torch.nn as nn
import torch

blu = '#206b99'
gren = '#b4ecb4'
purp =  '#A16E83'

real = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/normal2.npy")
#fake = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/gan2.npy")

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

Z = 3
N = 3
gen = Generator(Z, N)
gen.load_state_dict(torch.load('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Trained GANs/gan6coord.h5'))

latent_vector = torch.empty(len(real), Z).normal_(0, 1).float()
#test = torch.from_numpy(latent_vector)
fake = gen(latent_vector).detach().numpy()

print(np.shape(fake), np.shape(real))


real0,real1,real2,real3,real4,real5,fake0,fake1,fake2,fake3,fake4,fake5 = [],[],[],[],[],[],[],[],[],[],[],[]
for i in range(10000):
    real0.append(real[i][0])
    real1.append(real[i][1])
    real2.append(real[i][2])
    real3.append(real[i][3])
    real4.append(real[i][4])
    real5.append(real[i][5])

    fake0.append(fake[i][0])
    fake1.append(fake[i][1])
    fake2.append(fake[i][2])
    fake3.append(fake[i][3])
    fake4.append(fake[i][4])
    fake5.append(fake[i][5])


# 6 distribution plots for each independend coordinate3
plt.figure()

#plt.figure()
plt.title("x1")
sns.distplot(real0, hist=False, rug=False,label='Real', color=blu)
sns.distplot(fake0, hist=False, rug=False,label='Fake', color=purp)
plt.text(0.85,10,'$x_1$')
plt.legend()

#plt.figure()
plt.title("y1")
sns.distplot(real1, hist=False, rug=False, color=blu, kde_kws={'linestyle':'--'}) #,label='real y1'
sns.distplot(fake1, hist=False, rug=False, color=purp, kde_kws={'linestyle':'--'}) #,label='fake y1'
plt.text(1.1,9,'$y_1$')
plt.legend()

#plt.figure()
plt.title("x2")
sns.distplot(real2, hist=False, rug=False, color=blu) #,label='real x2'
sns.distplot(fake2, hist=False, rug=False, color=purp) #,label='fake x2'
plt.text(0.5,10,'$x_2$')
plt.legend()

#plt.figure()
plt.title("y2")
sns.distplot(real3, hist=False, rug=False, color=blu) #,label='real y2'
sns.distplot(fake3, hist=False, rug=False, color=purp) #,label='fake y2'
plt.text(2.25,7.5,'$y_2$')
plt.legend()

#plt.figure()
plt.title("x3")
sns.distplot(real4, hist=False, rug=False, color=blu, kde_kws={'linestyle':'--'}) #,label='real x3'
sns.distplot(fake4, hist=False, rug=False, color=purp, kde_kws={'linestyle':'--'}) #,label='fake x3',
plt.text(1.6,9,'$x_3$')
plt.legend()

#plt.figure()
plt.title("y3")
sns.distplot(real5, hist=False, rug=False, color=blu) #,label='real y3'
sns.distplot(fake5, hist=False, rug=False, color=purp) #,label='fake y3'
plt.text(1.9,9,'$y_3$')
plt.legend()

plt.xlabel('Coordinates $x_i$')
plt.ylabel('Frequency')
plt.title('Comparing distributions $p_g(z)$ and $p_d(x)$')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/gan6distrib.eps', bbox_inches='tight')

# Show the fake data

plt.figure()
plt.title("Real and Fake triangles")

c,b = [1, 3, 5], [0, 2, 4]

for i in range (0,20):
    FAKE = fake.tolist()
    samplefake = FAKE[random.randint(0,128 - 1)]
    xcoord_f = itemgetter(*b)(samplefake)
    ycoord_f = itemgetter(*c)(samplefake)
    plt.plot(xcoord_f, ycoord_f, 'ro', label = ('fake'))

    REAL = real.tolist()
    samplereal = REAL[random.randint(0,128)]
    xcoord_r = np.array(itemgetter(0,2,4)(samplereal))
    ycoord_r = np.array(itemgetter(1,3,5)(samplereal))
    plt.plot(xcoord_r, ycoord_r, 'bo', label = ('real'))

    i += 1
plt.show()
