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
N_star = 300
x_star = np.linspace(-6, 6, N_star).reshape(-1, 1)
D = 1
u_star = x_star**3

N = 32
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
# idx = np.array([26, 23, 4, 3, 27, 64, 58, 30, 18, 16, 2, 31, 65, 15, 11, 17, 57, 28, 34, 50])
idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        91, 100, 188,   8,  63,  98,  78])
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
noise_std = 9
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

#%% Model creation
layers = [1, 20, 20, D]
batch_size = N
num_batches = N / batch_size
klw = 1.0 / num_batches
model = BayesianNeuralNetwork(layers, 0.05, klw=klw,
                              pi_1=2., pi_2=0.1,
                              adv_eps=None, norm=NORM_NONE)
epochs = 7000
logger = Logger(epochs, frequency=100)
logger.set_val_err_fn(lambda: {})
model.fit(x_train, u_train, epochs, logger)

#%% Predictions and plotting
y_pred_list = []
u_pred, u_pred_var = model.predict(x_star, samples=100)
u_pred_sig = np.sqrt(u_pred_var)
lower = u_pred - 2 * u_pred_sig
upper = u_pred + 2 * u_pred_sig

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_star.ravel(), upper.ravel(), lower.ravel(), 
                 facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
plt.plot(x_star, u_pred, "b-", label=r"$\hat{u}_*(x)$")
plt.scatter(x_train, u_train, c="r", label=r"$u_T(x)$")
plt.plot(x_star, u_star, "r--", label=r"$u_*(x)$")
plt.xlabel("$x$")
# plt.show()
savefig(os.path.join("results", ""))


#%% Predictions and plotting
# fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
# plt.plot(x_star, u_star, "r--", label=r"$u_*(x)$")
# plt.scatter(x_train, u_train, c="r", label=r"$u_T(x)$")
# for i in range(3):
#     u_dist = model.predict_dist(x_star)
#     u_pred = u_dist.mean().numpy()
#     u_pred_sig = u_dist.stddev().numpy()
#     lower = u_pred - 2 * u_pred_sig
#     upper = u_pred + 2 * u_pred_sig

#     # plt.fill_between(x_star.ravel(), upper.ravel(), lower.ravel(), 
#                     # facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
#     plt.plot(x_star, upper, "g-", label=r"$\hat{u}_*(x)$")
#     plt.plot(x_star, lower, "g-", label=r"$\hat{u}_*(x)$")
#     plt.plot(x_star, u_pred, "b-", label=r"$\hat{u}_*(x)$")
#     plt.xlabel("$x$")
# plt.show()