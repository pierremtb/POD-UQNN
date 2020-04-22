"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

from poduqnn.custombnn import NORM_MEANSTD

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
HP["eps"] = 1e-6
HP["n_L"] = 0
HP["x_noise"] = 0.
HP["u_noise"] = 0.
# Train/val split
HP["train_val"] = (.8, .2)
# DeepNN Topology
HP["h_layers"] = [40, 40]
# Setting up TF SGD-based optimizer
HP["epochs"] = 200000
HP["lr"] = 0.003
HP["activation"] = "relu"
HP["pi_0"] = 0.5
HP["pi_1"] = 2.5
HP["pi_2"] = 0.1
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000
