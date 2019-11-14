"""Default hyperparameters for 1D Shekel Equation."""

import numpy as np
import tensorflow as tf


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
HP["n_s"] = 1000
# POD stopping param
HP["eps"] = 1e-10
HP["eps_init"] = 1e-10
# Train/val split
HP["train_val_ratio"] = 0.5
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Batch size for mini-batch training (0 means full-batch)
HP["batch_size"] = 0
# Setting up TF SGD-based optimizer
HP["epochs"] = 70000
HP["lr"] = 0.003
HP["decay"] = 0.
HP["b1"] = 0.9
HP["eps"] = 1e-10
HP["lambda"] = 1e-6
# Frequency of the logger
HP["log_frequency"] = 1000
# Non-spatial params
bet = 1/10 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
gam = 1. * np.array([4, 1, 8, 6, 3, 2, 5, 8, 6, 7])
mu_mean = np.hstack((bet, gam))
HP["mu_min"] = (mu_mean * (1 - np.sqrt(3)/10)).tolist()
HP["mu_max"] = (mu_mean * (1 + np.sqrt(3)/10)).tolist()


np.random.seed(1111)
tf.random.set_seed(1111)
