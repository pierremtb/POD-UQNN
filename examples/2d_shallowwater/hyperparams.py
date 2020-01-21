"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
import tensorflow as tf

from podnn.advneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


HP = {}
HP["mesh_idx"] = ([0], [1, 2], [4, 6, 7])
# Dimension of u(x, t, mu)
HP["n_v"] = len(HP["mesh_idx"][2])
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 300
# HP["n_s"] = 100
# POD stopping param
HP["eps"] = 0
HP["n_L"] = 10
# Train/val split
HP["train_val_test"] = (3/5, 1/5, 1/5)
# Deep NN hidden layers topology
HP["h_layers"] = [128, 128, 128]
# Setting up TF SGD-based optimizer
HP["n_M"] = 5
HP["epochs"] = 30000
HP["lr"] = 0.001
HP["lambda"] = 0.
HP["adv_eps"] = 1e-2
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
