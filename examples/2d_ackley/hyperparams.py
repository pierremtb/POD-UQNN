"""Default hyperparameters for 2D Ackley Equation."""

import numpy as np
import tensorflow as tf


HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 1
# Space
HP["n_x"] = 400
HP["x_min"] = -5
HP["x_max"] = +5.
HP["n_y"] = 400
HP["y_min"] = -5.
HP["y_max"] = +5.
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 3 * 100
HP["n_s_hifi"] = int(1e6)
# POD stopping param
HP["eps"] = 1e-10
# Train/val split
HP["train_val_test"] = (1/3, 1/3, 1/3)
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Setting up TF SGD-based optimizer
HP["epochs"] = 111000
HP["lr"] = 0.001
HP["lambda"] = 1e-2
# Frequency of the logger
HP["log_frequency"] = 1000
# Non-spatial params
HP["mu_min"] = [-1., -1., -1.]
HP["mu_max"] = [+1., +1., +1.]


np.random.seed(1111)
tf.random.set_seed(1111)
