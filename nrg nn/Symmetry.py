 #!/usr/local/bin/python3
 
"""
 This script is designed to test Networks 1, 2, and 3 on their ability to predict the energy for a molecule that has been translated or rotated. Physically invariant i.e. a symmetry of the Lagrangian, but now further from the data in the Training set
"""

import numpy as np
from keras.models import load_model
import random

net1 = load_model('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Networks/model1.h5')
net2 = load_model('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Networks/model2.h5')
net3 = load_model('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Networks/model3.h5')

datalist1 = np.load("datalist1.npy")
datalist2 = np.load("datalist2.npy")
datalist3 = np.load("datalist3.npy")

nrglist1 = np.load("nrglist1.npy")
nrglist2 = np.load("nrglist2.npy")
nrglist3 = np.load("nrglist3.npy")

# random point from test set (test sets were defined from 6000:10000
number = random.randint(6000,10000)

# Network 1 prediction
x = datalist1[number]
print(x)
x1 = np.reshape(x, (1,3))
print(x1)
y1 = nrglist1[number]

#x = [1.1224620482783192, 0.561231024147827, 0.9720806486538931]
#x1 = np.reshape(x, (1,3))

pred1 = net1.predict(x1, verbose =  1).flatten()

print('FIRST NETWORK: Prediction', pred1, 'Correct Value', y1, 'Error', pred1 - y1)


# Network 2 prediction
x = datalist2[number]
x2 = np.reshape(x, (1,6))
y2 = nrglist2[number]

pred2 = net2.predict(x2,  verbose = 1).flatten()

print('SECOND NETWORK: Prediction', pred2, 'Correct Value', y2, 'Error', pred2 - y2)

# Network 3 prediction
x = datalist3[number]
x3 = np.reshape(x, (1,3))
y3 = nrglist3[number]

pred3 = net3.predict(x3,  verbose = 1).flatten()

print('THIRD NETWORK: Prediction', pred3, 'Correct Value', y3, 'Error', pred3 - y3)
