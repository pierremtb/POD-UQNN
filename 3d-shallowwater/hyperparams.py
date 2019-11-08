"""Setup hyperparameters for 3D time-independant Shallow Water Equations."""

import numpy as np
import tensorflow as tf


HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 4
# Space
# HP["n_x"] = 256
# HP["n_y"] = 256
# HP["n_z"] = 256
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 100
# POD stopping param
HP["eps"] = 1e-10
HP["eps_init"] = 1e-10
# Train/val split
HP["train_val_ratio"] = 0.5
# Deep NN hidden layers topology
HP["h_layers"] = [64, 64, 64]
# Batch size for mini-batch training (0 means full-batch)
HP["batch_size"] = 0
# Setting up _structthe TF SGD-based optimizer
HP["epochs"] = 500000
# HP["epochs"] = 0
HP["lr"] = 0.001
HP["decay"] = 0.
HP["b1"] = 0.9
HP["eps"] = 1e-10
HP["lambda"] = 1e-6
# Frequency of the logger
HP["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
