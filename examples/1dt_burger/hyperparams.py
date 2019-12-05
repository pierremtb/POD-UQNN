"""Default hyperparameters for 1D time-dep Burgers Equation."""

import numpy as np
import tensorflow as tf


HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 1
# Space
HP["n_x"] = 256
HP["x_min"] = 0.
HP["x_max"] = 1.5
# Time
HP["n_t"] = 100
HP["t_min"] = 1.
HP["t_max"] = 5.
# Snapshots count
HP["n_s"] = 300
HP["n_s_hifi"] = int(1e3)
# POD stopping param
HP["eps"] = 1e-10
HP["eps_init"] = 1e-10
# HP["eps_init"] = None
# Train/val split
HP["train_val_test"] = (1/3, 1/3, 1/3)
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Setting up _structthe TF SGD-based optimizer
HP["epochs"] = 80000
HP["lr"] = 0.002
HP["lambda"] = 1e-4
# Frequency of the logger
HP["log_frequency"] = 1000
# Burgers params
HP["mu_min"] = [0.001]
HP["mu_max"] = [0.0100]

np.random.seed(1111)
tf.random.set_seed(1111)
