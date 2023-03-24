 #!/usr/lovanilla/bin/python3

import numpy as np
import  math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')
import torch.nn as nn
import torch
from matplotlib.colors import LinearSegmentedColormap

e=1
R0=10

blu = '#206b99'
gren = 'green'
purp =  '#A16E83'

vanilla = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/vanilla.npy")
wasser = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/wasser.npy")
real = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/uniform1.npy")

real0,real1,real2,was0,was1,was2,van0,van1,van2 = [],[],[],[],[],[],[],[],[]
for i in range(10):
    van0.append(vanilla[i][0])
    van1.append(vanilla[i][1])
    van2.append(vanilla[i][2])

    was0.append(wasser[i][0])
    was1.append(wasser[i][1])
    was2.append(wasser[i][2])

    real0.append(real[i][0])
    real1.append(real[i][1])
    real2.append(real[i][2])

plt.figure()
plt.title("wasser x2")
plt.xlabel('Coordinates $x_i$')
plt.ylabel('Frequency')
sns.distplot(van0, hist=False, rug=False,label='Vanilla', color=blu) #vanilla
sns.distplot(was0, hist=False, rug=False,label='Wasserstein',  color=purp)
sns.distplot(real0, hist=False, rug=False,label='Real',color=gren)
plt.text(0.95,8, '$x_2$')
plt.legend()


#plt.figure()
plt.title("wasser x3")
sns.distplot(van1, hist=False, rug=False, color=blu) #norm ,label='Real'
sns.distplot(was1, hist=False, rug=False, color=purp)
sns.distplot(real1, hist=False, rug=False,color=gren) #,label='Fake'color='red'
plt.text(0.85,7.3,'$x_3$')
plt.legend()

#plt.figure()
plt.title("wasser y3")
sns.distplot(van2, hist=False, rug=False, color=blu,kde_kws={'linestyle':'--'})
sns.distplot(was2, hist=False, rug=False,color=purp,kde_kws={'linestyle':'--'})
sns.distplot(real2, hist=False, rug=False, color=gren,kde_kws={'linestyle':'--'})
plt.text(1.2,7.5, '$y_3$')

plt.title('Comparing distributions $p_g(z)$ and $p_d(x)$')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/gan3distriution.eps', bbox_inches = 'tight')
plt.legend()

#kde_kws={'linestyle':'--'}

Ev, Ew = np.zeros(10000),np.zeros(10000)

for i in range (len(wasser)):
#    x2van.append(vanilla[i][0])
#    x3van.append(vanilla[i][1])
#    y3van.append(vanilla[i][2])
#
#    x2was.append(wasser[i][0])
#    x3was.append(wasser[i][1])
#    y3was.append(wasser[i][2])

    R12v = vanilla[i][0]
    R13v = math.sqrt(vanilla[i][1]**2 + vanilla[i][2]**2)
    R23v = math.sqrt((vanilla[i][1]-vanilla[i][0])**2 + vanilla[i][2]**2)

    R12w = wasser[i][0]
    R13w = math.sqrt(wasser[i][1]**2 + wasser[i][2]**2)
    R23w = math.sqrt((wasser[i][1]-wasser[i][0])**2 + wasser[i][2]**2)

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
