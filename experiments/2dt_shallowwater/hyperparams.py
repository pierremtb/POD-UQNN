"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
# import tensorflow as tf

from lib.varneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


HP = {}
HP["mesh_idx"] = ["eta"]
HP["mu_idx"] = [2]
# Dimension of u(x, t, mu)
HP["n_v"] = len(HP["mesh_idx"])
# Time
HP["n_t"] = 10
HP["d_t"] = 3
# Snapshots count
HP["n_s"] = 90
HP["n_s_tst"] = 10
# POD stopping param
HP["eps"] = 1e-6
HP["n_L"] = 0
# Train/val split
HP["train_val"] = (.8, .2)
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Setting up TF SGD-based optimizer
HP["n_M"] = 5
HP["epochs"] = 33000
HP["lr"] = 0.001
HP["lambda"] = 0.01
HP["adv_eps"] = 0.1
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000

# np.random.seed(1111)
# tf.random.set_seed(1111)
