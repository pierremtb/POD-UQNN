"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
import tensorflow as tf


HP = {}
HP["mesh_x_idx"] = ([0], [1, 2], [4,6,7])
# Dimension of u(x, t, mu)
HP["n_v"] = len(HP["mesh_x_idx"][2])
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 100
# POD stopping param
HP["eps"] = 1e-10
# Train/val split
HP["train_val_ratio"] = 0.99
# Deep NN hidden layers topology
HP["h_layers"] = [128, 128, 128]
# Batch size for mini-batch training (0 means full-batch)
HP["batch_size"] = 0
# Setting up _structthe TF SGD-based optimizer
# HP["epochs"] = 2000000
# HP["epochs"] = 20000
HP["epochs"] = 300000
HP["lr"] = 0.00001
HP["lambda"] = 1e-2
# HP["lambda"] = 0.
# Frequency of the logger
HP["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
