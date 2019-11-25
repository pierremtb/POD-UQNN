"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
import tensorflow as tf


HP = {}
HP["mesh_idx"] = ([0], [1, 2], [4, 6, 7])
# Dimension of u(x, t, mu)
HP["n_v"] = len(HP["mesh_idx"][2])
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 300
# POD stopping param
HP["eps"] = 1e-10
# Train/val split
HP["train_val_ratio"] = 0.7
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64]
# Setting up the TF SGD-based optimizer
# HP["epochs"] = 2000000
# HP["epochs"] = 40000
HP["epochs"] = 5000
HP["lr"] = 1e-1
HP["decay"] = 1e-2
# HP["lr"] = 0.001
# HP["decay"] = 0.
HP["lambda"] = 1e-6
# HP["lambda"] = 0.
# Frequency of the logger
HP["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
