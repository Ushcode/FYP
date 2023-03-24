 #!/usr/local/bin/python3

import numpy as np
import random
from operator import itemgetter
import matplotlib.pyplot as plt
plt.style.use('ggplot')

six_coord = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/uniform2.npy")
N=3 # atoms

def distance(data): # returns data as a list of interatomic distances
new = np.zeros((len(data), N))
for i in range(len(data)):
        translated[i][0] = math.sqrt((data[i][2]-data[i][0])**2+(data[i][3]-data[i][1])**2) #R12
        translated[i][1] = math.sqrt((data[i][4]-data[i][0])**2+(data[i][5]-data[i][1])**2) #R13
        translated[i][2] = math.sqrt((data[i][4]-data[i][2])**2+(data[i][5]-data[i][3])**2) #R23
     
return new

new = distance(six_coord)


