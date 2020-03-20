"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from poduqnn.logger import Logger
from poduqnn.custombnn import NORM_NONE, NORM_MEANSTD
from poduqnn.custombnn import BayesianNeuralNetwork
from poduqnn.plotting import figsize

#%% Datagen
N_star = 100
x_star = np.linspace(-6, 6, N_star).reshape(-1, 1)
D = 1
u_star = x_star**3

N = 32
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
# idx = np.array([26, 23, 4, 3, 27, 64, 58, 30, 18, 16, 2, 31, 65, 15, 11, 17, 57, 28, 34, 50])
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
noise_std = 5
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

#%% Model creation
layers = [1, 20, 20, D]
batch_size = N
num_batches = N / batch_size
klw = 1.0 / num_batches
model = BayesianNeuralNetwork(layers, 0.02, klw=klw, soft_0=1.,
                              adv_eps=None, norm=NORM_NONE)
epochs = 5000
logger = Logger(epochs, frequency=100)
logger.set_val_err_fn(lambda: {})
model.fit(x_train, u_train, epochs, logger)

#%% Predictions and plotting
y_pred_list = []

for i in range(100):
    y_pred = model.model.predict(x_star)
    y_pred_list.append(y_pred)
    
y_preds = np.concatenate(y_pred_list, axis=1)

u_pred = np.mean(y_preds, axis=1)
u_pred_sig = np.std(y_preds, axis=1)
lower = u_pred - 2 * u_pred_sig
upper = u_pred + 2 * u_pred_sig

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_star.ravel(), upper, lower, 
                 facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
plt.plot(x_star, u_pred, label=r"$\hat{u}_*(x)$")
plt.scatter(x_train, u_train, c="r", label=r"$u_T(x)$")
plt.plot(x_star, u_star, "r--", label=r"$u_*(x)$")
plt.xlabel("$x$")
plt.show()
