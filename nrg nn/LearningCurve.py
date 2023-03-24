 #!/usr/bin/python 

"""
This script is designed to retrain the model with increasing number of points in the training set. This is the learning curve
"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
import math

datalist = np.load("datalist.npy")
nrglist = np.load("nrglist.npy")

# Normalise input vector
x_mu = np.mean(datalist, axis=0)
x_std = np.std(datalist, axis=0)
datalist_NORM = (datalist - x_mu)/x_std

#normalise energies
y_mu = np.mean(nrglist)
y_std = np.std(nrglist)
nrglist_NORM = (nrglist - y_mu)/y_std

#Split into training, validation and test sets

    
foo = []
bar = []
foobar = []



for i in range (10, 120, 8):
    datalist_train = datalist_NORM[:i]
    datalist_val = datalist_NORM[6000:8000]
    datalist_test = datalist_NORM[8000:10000]

    nrglist_train = nrglist_NORM[:i]
    nrglist_val = nrglist_NORM[6000:8000]
    nrglist_test = nrglist_NORM[8000:10000]

    #Create model

    #Create model
    model = Sequential([
        Dense(64, input_shape = (3,)),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(1),
    ])

    # Compile model
    model.compile(optimizer = 'adam', loss = 'mse')

    callbacks = [EarlyStopping(patience=10, verbose=1), ModelCheckpoint(filepath='/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Networks/model.h5', save_best_only=True, verbose=1)]

    #print (model.summary())
    # Save the model with the modelcheckpoint callback (callbacks are fns carried out during training to monitor progress

    # Train
    epochs = 300

    history = model.fit(x=datalist_train, y=nrglist_train, epochs=epochs, validation_data=(datalist_val, nrglist_val), batch_size=32,  verbose = 1, callbacks = callbacks)

    pred_test = model.predict(datalist_test).flatten()
    pred_train = model.predict(datalist_train).flatten()
    mae_error = np.mean(np.abs(pred_test-nrglist_test)) # mean absolute error
    mae_error_train = np.mean(np.abs(pred_train-nrglist_train))
    print("Generalization Error (MAE): %f" % mae_error)
    mae_error = np.mean(np.abs(pred_test-nrglist_test)) # mean absolute error
    
    foo.append(i)
    bar.append(mae_error)
    foobar.append(mae_error_train)
   
    
plt.figure()
plt.title('Training and Test Error vs. Size of Training Set')
plt.xlabel('Number of points in training set')
plt.ylabel('Error (MAE) ')
plt.plot(foo, bar)
plt.plot(foo, foobar)
plt.legend(['test', 'train'], loc='upper right')
plt.show()

"""
# plot data
plt.figure()
p_min = np.min([pred_test, nrglist_test])
p_max = np.max([pred_test, nrglist_test])
plt.plot([p_min,p_max], [p_min,p_max], linestyle='dashed', color='k')

plt.scatter(pred_test, nrglist_test)
plt.xlabel('Prediction')
plt.ylabel('True')
plt.title('Test Set Predictions, MAE=%.3f' % mae_error)
plt.show()

    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png')

"""

