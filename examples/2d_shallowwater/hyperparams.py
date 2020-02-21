"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
import tensorflow as tf

from podnn.tfpbayesneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


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
HP["train_val"] = (3/5, 1/5, 1/5)
# DeepNN Topology
HP["h_layers"] = [50, 50, 50, 50]
HP["h_layers_t"] = [50, 50]
# Setting up TF SGD-based optimizer
HP["epochs"] = 200000
HP["lr"] = 0.001
HP["lambda"] = 1.5
HP["beta"] = 0.
HP["k1"] = 1
HP["k2"] = 5
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
