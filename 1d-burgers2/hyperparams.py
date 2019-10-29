import numpy as np
import tensorflow as tf


hp = {}
# Space
hp["n_x"] =  [256]
hp["x_min"] = [0.]
hp["x_max"] = [1.5]
# Time
hp["n_t"] = 100
hp["t_min"] = 1.
hp["t_max"] = 5.
# Snapshots count
hp["n_s"] = 2 * 100
# POD stopping param
hp["eps"] = 1e-10
hp["eps_init"] = 1e-10
# Train/val split
hp["train_val_ratio"] = 0.5
# Deep NN hidden layers topology
hp["h_layers"] = [64, 64]
# Batch size for mini-batch training (0 means full-batch)
hp["batch_size"] = 0
# Setting up the TF SGD-based optimizer
hp["tf_epochs"] = 100000
hp["tf_lr"] = 0.002
hp["tf_decay"] = 0.
hp["tf_b1"] = 0.9
hp["tf_eps"] = None
hp["lambda"] = 1e-4
# Frequency of the logger
hp["log_frequency"] = 1000
# Burgers params
hp["mu_min"] = [0.001]
hp["mu_max"] = [0.0100]
# hp["mu_min"] = 0.01/np.pi * (1 - np.sqrt(3)/10) 
# hp["mu_max"] = 0.01/np.pi * (1 + np.sqrt(3)/10) 

np.random.seed(1111)
tf.random.set_seed(1111)
