"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
import tensorflow as tf


HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 3
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 100
# POD stopping param
HP["eps"] = 1e-10
# Train/val split
HP["train_val_ratio"] = 0.5
# Deep NN hidden layers topology
HP["h_layers"] = [128, 128, 128]
# Batch size for mini-batch training (0 means full-batch)
HP["batch_size"] = 0
# Setting up _structthe TF SGD-based optimizer
HP["epochs"] = 2000000
HP["lr"] = 0.001
HP["lambda"] = 1e-2
# HP["lambda"] = 0.
# Frequency of the logger
HP["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
