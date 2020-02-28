"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
# import tensorflow as tf

# from podnn.varneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


HP = {}
HP["mesh_idx"] = ([0], [1, 2], [4, 6, 7])
HP["mu_idx"] = [2]
# Dimension of u(x, t, mu)
HP["n_v"] = len(HP["mesh_idx"][2])
# Time
HP["n_t"] = 10
HP["d_t"] = 1
# Snapshots count
HP["n_s"] = 20
HP["n_s_tst"] = 3
# POD stopping param
HP["eps"] = 0
HP["n_L"] = 10
# Train/val split
HP["train_val"] = (.8, .2)
# Deep NN hidden layers topology
HP["h_layers"] = [128, 128, 128]
# Setting up TF SGD-based optimizer
HP["n_M"] = 5
HP["epochs"] = 40000
HP["lr"] = 0.01
HP["lambda"] = 0.001
HP["adv_eps"] = 0.
HP["norm"] = "none"
# Frequency of the logger
HP["log_frequency"] = 1000

# np.random.seed(1111)
# tf.random.set_seed(1111)
