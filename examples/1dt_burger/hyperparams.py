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
HP["n_s"] = 2 * 100
HP["n_s_hifi"] = 1e5
# POD stopping param
HP["eps"] = 1e-10
# HP["eps_init"] = 1e-10
HP["eps_init"] = None
# Train/val split
HP["train_val_ratio"] = 0.5
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Setting up _structthe TF SGD-based optimizer
# HP["epochs"] = 30000
HP["epochs"] = 1000
HP["lr"] = 0.002
HP["lambda"] = 1e-4
# Frequency of the logger
HP["log_frequency"] = 1000
# Burgers params
HP["mu_min"] = [0.001]
HP["mu_max"] = [0.0100]
# HP["mu_min"] = 0.01/np.pi * (1 - np.sqrt(3)/10) 
# HP["mu_max"] = 0.01/np.pi * (1 + np.sqrt(3)/10) 

np.random.seed(1111)
tf.random.set_seed(1111)
