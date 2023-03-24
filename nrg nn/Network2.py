 #!/usr/bin/python

"""
This Network is trained with 10,000 points generated by TASK1. The data is NORMALISED before splitting into training, val, test sets.

Second network, uses data with 6 dof instead of 3 as in Net1
"""
plum = '#A16E83'
blu = '#206B99'

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import math

datalist = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/normal2.npy") # normal or uniform
nrglist = np.load("/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Data/nrg_n2.npy")

# Normalise input vector
x_mu = np.mean(datalist, axis=0)
x_std = np.std(datalist, axis=0)
datalist_NORM = (datalist - x_mu)/x_std

#normalise energies
y_mu = np.mean(nrglist)
y_std = np.std(nrglist)
nrglist_NORM = (nrglist - y_mu)/y_std

#Split into training, validation and test sets

datalist_train = datalist_NORM[:6000]
datalist_val = datalist_NORM[6000:8000]
datalist_test = datalist_NORM[8000:10000]

nrglist_train = nrglist_NORM[:6000]
nrglist_val = nrglist_NORM[6000:8000]
nrglist_test = nrglist_NORM[8000:10000]

#Create model
model = Sequential([
    Dense(64, input_shape = (6,)),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(1),
])

# Compile model
model.compile(optimizer = 'adam', loss = 'mse')

callbacks = [EarlyStopping(patience=100, verbose=1), ModelCheckpoint(filepath='/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/Trained Networks/model2.h5', save_best_only=True, verbose=1)]


# Train
epochs = 300

history = model.fit(x=datalist_train, y=nrglist_train, epochs=epochs, validation_data=(datalist_val, nrglist_val), batch_size=32,  verbose = 1, callbacks = callbacks)

#  Test Error
pred_test = model.predict(datalist_test).flatten()
mae_error = np.mean(np.abs(pred_test-nrglist_test)) # mean absolute error
print("Generalization Error (MAE): %f" % mae_error)

# plot loss
plt.figure()
plt.plot(history.history['loss'], label='Train', color=plum)
plt.plot(history.history['val_loss'], label='Validation', color=blu)
plt.title('Model loss (MSE)')
plt.ylabel('Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/loss2.eps', bbox_inches='tight')

# plot predictions vs true values
plt.figure()
p_min = np.min([pred_test, nrglist_test])
p_max = np.max([pred_test, nrglist_test])
plt.plot([p_min,p_max], [p_min,p_max], linestyle='dashed', color='k')

plt.scatter(pred_test, nrglist_test, color = plum)
plt.xlabel('Prediction')
plt.ylabel('True')
plt.title('Test Set Predictions, MAE=%.3f' % mae_error)
plt.savefig('/Users/Oisin/Documents/Theoretical Physics/PROJECT/CODE/figs/prediction2.eps', bbox_inches='tight')
plt.show()
