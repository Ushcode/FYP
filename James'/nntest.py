#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

N = 2000  # number of points in dataset
X = np.random.uniform(-1,1,(N, 2))  # generate input randomly over 2D space

def F(x, y):
    # the function that we want to learn
    return y*np.sin(4*x) + 3*np.cos(1.1*x*y+2.3)

y = F(X[:, 0], X[:, 1])  # targets



# 3d plot the dataset
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs=X[:,0], ys=X[:,1], zs=y, zdir='z', s=20, c=None, depthshade=True)
# plt.show()


# split into training, validation & test sets
X_train, X_val, X_test = X[:1000], X[1000:1400], X[1400:2000]
y_train, y_val, y_test = y[:1000], y[1000:1400], y[1400:2000]


# create model
nn = Sequential()
nn.add(Dense(input_shape=(2,), units=64, activation='relu'))
nn.add(Dense(units=32, activation='relu'))
nn.add(Dense(units=1))  # no activation on output for regression
print(nn.summary())


callbacks = [EarlyStopping(patience=10, verbose=1), ModelCheckpoint(filepath='SavedNets/nntest.h5', save_best_only=True, verbose=1)]
nn.compile(optimizer='adam', loss='mse')


# train
epochs = 50
history = nn.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)

# plot history
plt.figure()
plt.plot(range(1, len(history.history['loss'])+1), history.history['loss'], label='Train')
plt.plot(range(1, len(history.history['val_loss'])+1), history.history['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.yscale('log')

# load the best model
nn_best = load_model('SavedNets/nntest.h5')

# compute generalization error using the test set
pred_test = nn.predict(X_test).flatten()
mae_error = np.mean(np.abs(pred_test-y_test)) # mean absolute error
print("Generalization Error (MAE): %f" % mae_error)

# plot data
plt.figure()
p_min = np.min([pred_test, y_test])
p_max = np.max([pred_test, y_test])
plt.plot([p_min,p_max], [p_min,p_max], linestyle='dashed', color='k')

plt.scatter(pred_test, y_test)
plt.xlabel('Prediction')
plt.ylabel('True')
plt.title('Test Set Predictions, MAE=%.3f' % mae_error)
plt.show()


