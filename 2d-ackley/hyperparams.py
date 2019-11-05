import numpy as np
import tensorflow as tf


hp = {}
# Dimension of u(x, t, mu)
hp["n_v"] = 1
# Space
hp["n_x"] = 400
hp["x_min"] = -5
hp["x_max"] = +5.
hp["n_y"] = 400
hp["y_min"] = -5.
hp["y_max"] = +5.
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
# Setting up TF SGD-based optimizer
hp["epochs"] = 100000
hp["lr"] = 0.002
hp["eps"] = 1e-10
hp["lambda"] = 1e-4
# Frequency of the logger
hp["log_frequency"] = 1000
# Non-spatial params
hp["mu_min"] = [-1., -1., -1.]
hp["mu_max"] = [+1., +1., +1.]


np.random.seed(1111)
tf.random.set_seed(1111)
