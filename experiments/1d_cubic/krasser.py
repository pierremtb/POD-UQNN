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
from poduqnn.plotting import figsize, savefig

#%% Datagen
N_star = 300
x_star = np.linspace(-1.5, 1.5, N_star).reshape(-1, 1)
D = 1
noise_std = 1.
def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon
# u_star = f(x_star, noise_std)

N = 32
x_train = np.linspace(-0.5, 0.5, N).reshape(-1, 1)
u_train = f(x_train, sigma=noise_std)

#%% Model creation
layers = [1, 20, 20, D]
batch_size = N
num_batches = N / batch_size
klw = 1.0 / num_batches
model = BayesianNeuralNetwork(layers, 0.08, klw=klw,
                              pi_1=2., pi_2=0.1,
                              adv_eps=None, norm=NORM_NONE)
epochs = 1500
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
# plt.plot(x_star, u_star, "r--", label=r"$u_*(x)$")
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