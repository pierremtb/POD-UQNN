"""Default hyperparameters for 1D Shekel Equation."""

import sys
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

sys.path.append(os.path.join("..", ".."))
from podnn.tfpbayesneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 1
# Space
HP["n_x"] = 300
HP["x_min"] = 0.
HP["x_max"] = 10.
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 500
HP["n_s_tst"] = 300
# POD stopping param
HP["eps"] = 1e-3
HP["n_L"] = 0
HP["x_noise"] = 0.
HP["u_noise"] = 0.
# Train/val split
HP["train_val"] = (0.8, 0.2)
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64, 64]
# Setting up TF SGD-based optimizer
HP["epochs"] = 60000
HP["lr"] = 0.008
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 500
# Non-spatial params
# bet = 1/10 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
# gam = 1. * np.array([4, 1, 8, 6, 3, 2, 5, 8, 6, 7])
bet = 1/10 * np.array([1, 2, 2, 4, 4])
gam = 1. * np.array([4, 1, 8, 6, 3])
mu_mean = np.hstack((bet, gam))
HP["mu_min"] = (mu_mean * (1 - np.sqrt(3)/10)).tolist()
HP["mu_max"] = (mu_mean * (1 + np.sqrt(3)/10)).tolist()
HP["mu_min_out"] = (mu_mean * (1 - 1.5*np.sqrt(3)/10)).tolist()
HP["mu_max_out"] = (mu_mean * (1 + 1.5*np.sqrt(3)/10)).tolist()


# np.random.seed(1111)
# tf.random.set_seed(1111)

def u(X, _, mu):
    """The 1D-Shekel function."""
    x = X[0]
    sep = int(mu.shape[0] / 2)
    bet = mu[:sep]
    gam = mu[sep:]

    u_sum = np.zeros_like(x)
    for i in range(len(bet)):
        i_sum = (x - gam[i])**2
        u_sum += 1 / (bet[i] + i_sum)

    return u_sum.reshape((1, u_sum.shape[0]))