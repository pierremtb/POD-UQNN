"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

import numpy as np
import tensorflow as tf

from podnn.bayesianneuralnetwork import NORM_MINMAX, NORM_MEANSTD

HP = {}
HP["mesh_idx"] = ([0], [1, 2], [4, 6, 7])
# Dimension of u(x, t, mu)
HP["n_v"] = len(HP["mesh_idx"][2])
# Time
HP["n_t"] = 0
# Snapshots count
HP["n_s"] = 500
HP["n_s_tst"] = 300
# POD stopping param
HP["eps"] = 1e-4
HP["n_L"] = 0
HP["x_noise"] = 0.
HP["u_noise"] = 0.
# Train/val split
HP["train_val"] = (.8, .2)
# DeepNN Topology
HP["h_layers"] = [40, 40]
# Setting up TF SGD-based optimizer
HP["epochs"] = 100000
HP["lr"] = 0.003
HP["soft_0"] = 0.01
HP["sigma_alea"] = 200.
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000

np.random.seed(1111)
tf.random.set_seed(1111)
