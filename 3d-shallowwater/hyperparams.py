import numpy as np
import tensorflow as tf


hp = {}
# Dimension of u(x, t, mu)
hp["n_v"] = 4
# Space
# hp["n_x"] = 256
# hp["n_y"] = 256
# hp["n_z"] = 256
# Time
hp["n_t"] = 0
# Snapshots count
hp["n_s"] = 100
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
# hp["epochs"] = 30000
hp["epochs"] = 0
hp["lr"] = 0.01
hp["decay"] = 0.
hp["b1"] = 0.9
hp["eps"] = 1e-10
hp["lambda"] = 1e-4
# Frequency of the logger
hp["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
