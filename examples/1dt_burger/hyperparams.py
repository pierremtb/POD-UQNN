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
HP["n_s"] = 100
HP["n_s_tst"] = 100
# POD stopping param
HP["eps"] = 1e-10
HP["eps_init"] = 1e-10
# Train/val split
HP["train_val"] = (0.8, 0.2)
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Setting up _structthe TF SGD-based optimizer
HP["epochs"] = 45000
HP["epochs"] = 10000
HP["lr"] = 0.008
HP["lambda"] = 1e-8
# Frequency of the logger
HP["log_frequency"] = 1000
# Burgers params
HP["mu_min"] = [0.001]
HP["mu_max"] = [0.0100]
HP["mu_min_out"] = [0.0005]
HP["mu_max_out"] = [0.0105]

np.random.seed(1111)
tf.random.set_seed(1111)
