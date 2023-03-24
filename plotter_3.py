 #!/usr/lonormal/bin/python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')
import torch.nn as nn
import torch

blu = '#206b99'
gren = '#b4ecb4'
purp =  '#A16E83'

normal = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/normal1.npy")
uniform = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/uniform1.npy")
fake = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/gan1batch.npy")
wasser = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/wasser.npy")

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

Z = 4
gen = Generator(Z)
gen.load_state_dict(torch.load('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Trained GANs/gan3coord.h5'))

latent_vector = torch.empty(len(uniform), Z).normal_(0, 1).float()
#test = torch.from_numpy(latent_vector)
fake = gen(latent_vector).detach().numpy()

#Wassergan1UNIF.npy

norm0,norm1,norm2,uni0,uni1,uni2,fake0,fake1,fake2 = [],[],[],[],[],[],[],[],[]
for i in range(10000):
    norm0.append(normal[i][0])
    norm1.append(normal[i][1])
    norm2.append(normal[i][2])

    print(normal[i][0], normal[i][1], normal[i][2], "NORM")
    
#    uni0.append(uniform[i][0])
#    uni1.append(uniform[i][1])
#    uni2.append(uniform[i][2])
#
#    print(uniform[i][0], uniform[i][1], uniform[i][2], "UNI")
    
    fake0.append(fake[i][0])
    fake1.append(fake[i][1])
    fake2.append(fake[i][2])

plt.figure()
plt.title("uniform x2")
plt.xlabel('Coordinates $x_i$')
plt.ylabel('Frequency')
sns.distplot(norm0, hist=False, rug=False,label='Real', color=blu) #normal
#sns.distplot(uni0, hist=False, rug=False,label='uniform')
sns.distplot(fake0, hist=False, rug=False,label='Fake', color=purp)
plt.text(0.95,8, '$x_2$')
plt.legend()


#plt.figure()
plt.title("uniform x3")
sns.distplot(norm1, hist=False, rug=False, color=blu) #norm ,label='Real'
#sns.distplot(uni1, hist=False, rug=False,label='uniform')
sns.distplot(fake1, hist=False, rug=False, color=purp) #,label='Fake'
plt.text(0.63,8.5,'$x_3$')
plt.legend()

#plt.figure()
plt.title("uniform y3")
sns.distplot(norm2, hist=False, rug=False, color=blu) #norm ,label='Real'
#sns.distplot(uni2, hist=False, rug=False,label='uniform')
sns.distplot(fake2, hist=False, rug=False, color=purp) #label='Fake',
plt.text(1.13,8.5, '$y_3$')

plt.title('Comparing distributions $p_g(z)$ and $p_d(x)$')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/gan3distriution.eps', bbox_inches = 'tight')
plt.legend()

#kde_kws={'linestyle':'--'}


plt.show()
