#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:22:03 2019

@author: Aisling
"""
import numpy as np
import matplotlib.pyplot as py
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

x = np.load('m0_x.npy') 
y = np.load('m0_y.npy')


X_train, X_val, X_test = x[:6000], x[6000:8000], x[8000:10001]
y_train, y_val, y_test = y[:6000], y[6000:8000], y[8000:10001]

x_mu = np.mean(X_train, axis=0)
x_std = np.std(X_train, axis=0)

#take non normalised array without 0s, normalise, convert to list and put zeros back in then back to array
def normaliser(x_array):
    new = np.zeros(len(x_array))
    new = (x_array - x_mu )/x_std
    new1 = new.tolist()
    for element in new1:
        element.insert(0, 0.0)
        element.insert(0, 0.0)
        element.insert(3, 0.0)
    new2 = np.asarray(new1)
    return(new2)    

X_train_n = normaliser(X_train)

X_val_n = normaliser(X_val)

X_test_n = normaliser(X_test)

#normalise energies

y_mu = np.mean(y_train)
y_std = np.std(y_train)

m = np.array((y_mu, y_std))

np.save('mu_std_array',m)

y_train_n = (y_train - y_mu)/y_std
y_val_n = (y_val - y_mu)/y_std
y_test_n = (y_test - y_mu)/y_std

nn = Sequential()
nn.add(Dense(input_shape=(8,), units=64, activation='relu'))
nn.add(Dense(units=32, activation='relu'))
nn.add(Dense(units=1))  # no activation on output for regression
#print(nn.summary())

callbacks = [EarlyStopping(patience=10, verbose=1), ModelCheckpoint(filepath='nntest0.h5', save_best_only=True, verbose=1)]
nn.compile(optimizer='adam', loss='mse')

epochs = 500
history = nn.fit(x=X_train_n, y=y_train_n, epochs=epochs, batch_size=128, validation_data=(X_val_n, y_val_n), callbacks=callbacks)

# plot history
py.figure()
py.plot(range(1, len(history.history['loss'])+1), history.history['loss'], label='Train')
py.plot(range(1, len(history.history['val_loss'])+1), history.history['val_loss'], label='Validation')
py.xlabel('Epochs')
py.ylabel('Mean Squared Error (MSE)')
py.legend()
py.yscale('log')

# load the best model
nn_best = load_model('nntest0.h5')

# compute generalization error using the test set
pred_test = nn.predict(X_test_n).flatten()
mae_error = np.mean(np.abs(pred_test-y_test_n)) # mean absolute error
#print("Generalization Error (MAE): %f" % mae_error)

# plot data
py.figure()
p_min = np.min([pred_test, y_test_n])
p_max = np.max([pred_test, y_test_n])
py.plot([p_min,p_max], [p_min,p_max], linestyle='dashed', color='k')

py.scatter(pred_test, y_test_n)
py.xlabel('Prediction')
py.ylabel('True')
py.title('Test Set Predictions, MAE=%.3f' % mae_error)
py.show()
