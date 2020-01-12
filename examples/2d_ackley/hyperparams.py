"""Default hyperparameters for 2D Ackley Equation."""

import numpy as np
import tensorflow as tf

from podnn.advneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


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
HP["n_s_hifi"] = int(5e2)
# POD stopping param
HP["eps"] = 0
HP["n_L"] = 12
HP["x_noise"] = 0.
HP["u_noise"] = 0.
# Train/val split
HP["train_val_test"] = (3/5, 1/5, 1/5)
# DeepNN Topology
HP["h_layers"] = [50, 50, 50, 50]
HP["h_layers_t"] = [50, 50, 50]
# Setting up TF SGD-based optimizer
HP["epochs"] = 194000
HP["epochs"] = 0
HP["lr"] = 0.00005
HP["lambda"] = 1.5
HP["beta"] = 1e-6
HP["k1"] = 1
HP["k2"] = 25
HP["norm"] = NORM_MEANSTD
# HP["norm"] = NORM_CENTER
# Frequency of the logger
HP["log_frequency"] = 2000
# Non-spatial params
HP["mu_min"] = [-1., -1., -1.]
HP["mu_max"] = [+1., +1., +1.]


np.random.seed(1111)
tf.random.set_seed(1111)
