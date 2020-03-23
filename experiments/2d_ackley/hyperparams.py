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
HP["n_s"] = 500
HP["n_s_tst"] = 300
# POD stopping param
HP["eps"] = 1e-10
# Train/val split
HP["train_val"] = (0.8, 0.2)
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Setting up TF SGD-based optimizer
HP["epochs"] = 30000
HP["lr"] = 0.003
HP["lambda"] = 1e-2
# Frequency of the logger
HP["log_frequency"] = 1000
# Non-spatial params
HP["mu_min"] = [-1., -1., -1.]
HP["mu_max"] = [+1., +1., +1.]
HP["mu_min_out"] = [-2., -2., -2.]
HP["mu_max_out"] = [+2., +2., +2.]


np.random.seed(1111)
tf.random.set_seed(1111)
