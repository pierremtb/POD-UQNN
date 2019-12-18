"""Default hyperparameters for 1D Shekel Equation."""

import numpy as np
import tensorflow as tf
from podnn.advneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


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
HP["n_s"] = 300
# HP["n_s_hifi"] = int(1e5)
HP["n_s_hifi"] = int(1e4)
# POD stopping param
# HP["eps"] = 1e-10
HP["n_L"] = 20
HP["u_noise"] = 0.
# Train/val split
HP["train_val_test"] = (3/5, 1/5, 1/5)
# Deep NN hidden layers topology
HP["h_layers"] = [50, 50, 50, 50]
HP["h_layers_t"] = [50, 50, 50]
# Setting up TF SGD-based optimizer
HP["epochs"] = 8000
HP["lr"] = 0.001
HP["lambda"] = 1.5
HP["beta"] = 0
# HP["beta"] = 1e-2
HP["k1"] = 1
HP["k2"] = 5
HP["norm"] = NORM_MEANSTD
# HP["norm"] = NORM_CENTER
# HP["norm"] = NORM_NONE
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


np.random.seed(1111)
tf.random.set_seed(1111)
