"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

from poduqnn.varneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE


HP = {}
HP["mesh_idx"] = ["eta"]
HP["mu_idx"] = [2]
# Dimension of u(x, t, mu)
HP["n_v"] = len(HP["mesh_idx"])
# Time
HP["n_t"] = 100
HP["d_t"] = 0.1
HP["t_min"] = 0.
HP["t_max"] = 99.
# Snapshots count
HP["n_s"] = 50
HP["n_s_tst"] = 1
# POD stopping param
HP["eps"] = 1e-8
HP["n_L"] = 0
# Train/val split
HP["train_val"] = (.8, .2)
# Deep NN hidden layers topology
HP["h_layers"] = [140, 140]
# Setting up TF SGD-based optimizer
HP["n_M"] = 5
HP["epochs"] = 150000
HP["lr"] = 0.005
HP["lambda"] = 0.001
HP["adv_eps"] = 0.001
HP["soft_0"] = 0.01
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000
