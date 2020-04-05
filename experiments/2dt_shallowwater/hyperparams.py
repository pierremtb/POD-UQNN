"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

from poduqnn.custombnn import NORM_MEANSTD

HP = {}
HP["mesh_idx"] = ["eta"]
HP["mu_idx"] = [2]
# Dimension of u(x, t, mu)
HP["n_v"] = 1
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
# DeepNN Topology
HP["h_layers"] = [140, 140]
# Setting up TF SGD-based optimizer
HP["epochs"] = 200000
HP["lr"] = 0.003
HP["pi_1"] = 2.
HP["pi_2"] = 0.1
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000
