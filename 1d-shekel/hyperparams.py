import numpy as np
import tensorflow as tf


hp = {}
# Dimension of u(x, t, mu)
hp["n_v"] = 1
# Space
hp["n_x"] = 300
hp["x_min"] = 0.
hp["x_max"] = 10.
# Time
hp["n_t"] = 0
# Snapshots count
hp["n_s"] = 1000
# POD stopping param
hp["eps"] = 1e-10
hp["eps_init"] = 1e-10
# Train/val split
hp["train_val_ratio"] = 0.5
# Deep NN hidden layers topology
hp["h_layers"] = [64, 64]
# Batch size for mini-batch training (0 means full-batch)
hp["batch_size"] = 0
# Setting up _structthe TF SGD-based optimizer
hp["epochs"] = 70000
hp["lr"] = 0.003
hp["decay"] = 0.
hp["b1"] = 0.9
hp["eps"] = 1e-10
hp["lambda"] = 1e-6
# Frequency of the logger
hp["log_frequency"] = 1000
# Non-spatial params
bet = 1/10 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
gam = 1. * np.array([4, 1, 8, 6, 3, 2, 5, 8, 6, 7])
mu_mean = np.hstack((bet, gam))
hp["mu_min"] = mu_mean * (1 - np.sqrt(3)/10)
hp["mu_max"] = mu_mean * (1 + np.sqrt(3)/10)


np.random.seed(1111)
tf.random.set_seed(1111)
