import numpy as numpy
import tensorflow as tf


hp = {}
# Space (dx = 1/30, n_x = 10/dx)
hp["n_x"] = 300
# Snapshots count
hp["n_s"] = 1000
# POD stopping param
hp["eps"] = 1e-10
# Train/val split
hp["train_val_ratio"] = 0.7
# Deep NN hidden layers topology
hp["h_layers"] = [64, 64]
# Batch size for mini-batch training (0 means full-batch)
hp["batch_size"] = 0
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 70000
hp["tf_lr"] = 0.003
hp["tf_decay"] = 0.
hp["tf_b1"] = 0.9
hp["tf_eps"] = None
hp["lambda"] = 1e-6
# Frequency of the logger
hp["log_frequency"] = 1000
# Shekel params
hp["bet_count"] = 10
hp["gam_count"] = 10


np.random.seed(1111)
tf.random.set_seed(1111)
