"""Default hyperparameters for 1D time-dep Burgers Equation."""

import numpy as np
import tensorflow as tf
from podnn.advneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE

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
# Train/val split
HP["train_val_test"] = (3/5, 1/5, 1/5)
# DeepNN Topology
HP["h_layers"] = [50, 50, 50, 50]
HP["h_layers_t"] = [50, 50, 50]
# Setting up TF SGD-based optimizer
HP["epochs"] = 26000
HP["lr"] = 0.001
HP["lambda"] = 1.5
HP["beta"] = 1e-5
HP["k1"] = 1
HP["k2"] = 5
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 500
# Burgers params
HP["mu_min"] = [0.001]
HP["mu_max"] = [0.0100]

np.random.seed(1111)
tf.random.set_seed(1111)
