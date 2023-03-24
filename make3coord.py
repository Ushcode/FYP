 #!/usr/local/bin/python3
import numpy as np
import random
from operator import itemgetter
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plum = '#A16E83'
blu = '#206B99'
gren = '#b4ecb4'
red = '#F78888'

six_coord = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/uniform2.npy")
N = 3 # atoms in configuration space

def translate(data): # translate (x1, y1) to origin, eliminating these coordinates from the parameterisation space
    translated = np.zeros((len(data), 2*N))
    for i in range(len(data)):
            translated[i][0] = data[i][0] - data[i][0]
            translated[i][2] = data[i][2] - data[i][0]
            translated[i][4] = data[i][4] - data[i][0]
            translated[i][1] = data[i][1] - data[i][1]
            translated[i][3] = data[i][3] - data[i][1]
            translated[i][5] = data[i][5] - data[i][1]
    return translated

def rotate(data):
    rotated = np.zeros((len(data), 2*N))
    for i in range (len(data)):
        y2,x2 = data[i][5],data[i][4]
        angle = np.arctan(y2/x2)
        R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        p2, p3 = np.array([data[i][2], data[i][3]]), np.array([data[i][4], data[i][5]])
        p2, p3 = np.matmul(R, p2), np.matmul(R, p3)
        
        rotated[i][2] = p2[0]
        rotated[i][3] = p2[1]
        rotated[i][4] = p3[0]
        rotated[i][5] = p3[1]
    return rotated

for i in range(0,20): # plot new and old triangles
    old = (six_coord).tolist()
    sample6 = old[random.randint(0, len(six_coord))]
    xcoord_f = itemgetter(0,2,4)(sample6)
    ycoord_f = itemgetter(1,3,5)(sample6)
    plt.scatter(xcoord_f, ycoord_f, color=blu, marker = '.', label = ('six'))

    new = rotate(translate(six_coord)).tolist()
    sample3 = new[random.randint(0,len(six_coord))]
    xcoord_r = np.array(itemgetter(0,2,4)(sample3))
    ycoord_r = np.array(itemgetter(1,3,5)(sample3))
    plt.scatter(xcoord_r, ycoord_r, color = plum, marker='.', label = ('three'))

    i += 1

new = rotate(translate(six_coord))

#un, uny, deux, deuxy, trois, troisy = [],[],[],[],[],[]

three_coord = []
for i in range(len(six_coord)):
    three_coord.append([new[i][4], new[i][2], new[i][3]])
np.save("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/reduced", three_coord)

#for i in range (20):
#    un.append(0) # plot new triangles to make sure
#    uny.append(0)
#    deuxy.append(0)
#    deux.append(three_coord[i][0])
#    trois.append(three_coord[i][1])
#    troisy.append(three_coord[i][2])
#plt.figure()
#plt.title('randomly generated positions')
#plt.plot(un, uny, 'ro')
#plt.plot(deux, deuxy, 'bo')
#plt.plot(trois, troisy, 'go')

plt.title('Triangular Molecule representations')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/transformation.eps', bbox_inches = 'tight')
plt.show()
