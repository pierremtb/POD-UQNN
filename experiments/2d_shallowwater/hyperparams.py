"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
import tensorflow as tf

from poduqnn.varneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


HP = {}
HP["mesh_idx"] = ["h"]
HP["mu_idx"] = [0]
# Dimension of u(x, t, mu)
HP["n_v"] = 1
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 180
HP["n_s_tst"] = 20
# POD stopping param
HP["eps"] = 1e-10
HP["n_L"] = 0
# Train/val split
HP["train_val"] = (3/5, 1/5)
# Deep NN hidden layers topology
HP["h_layers"] = [40, 40]
# HP["h_layers"] = [128, 128, 128]
# Setting up TF SGD-based optimizer
HP["n_M"] = 5
# HP["epochs"] = 120000
HP["epochs"] = 50000
HP["lr"] = 0.01
# HP["lr"] = 0.003
HP["lambda"] = 1e-8
HP["soft_0"] = 1.
HP["adv_eps"] = 0.
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 5000

np.random.seed(1111)
tf.random.set_seed(1111)
