"""Default hyperparameters for 2D inviscid Shallow Water Equations."""

from poduqnn.custombnn import NORM_MEANSTD

HP = {}
HP["mesh_idx"] = ["eta"]
HP["mu_idx"] = [2]
# Dimension of u(x, t, mu)
HP["n_v"] = 1
# Time
HP["n_t"] = 101
HP["d_t"] = 0.3
HP["t_min"] = 0.
HP["t_max"] = 100.
# Snapshots count
HP["n_s"] = 98
HP["n_s_tst"] = 2
# POD stopping param
HP["eps"] = 1e-6
HP["eps_init"] = 1e-6
HP["n_L"] = 0
# Train/val split
HP["train_val"] = (.8, .2)
# DeepNN Topology
HP["h_layers"] = [128, 128, 128]
# Setting up TF SGD-based optimizer
HP["epochs"] = 120000
HP["lr"] = 0.003
HP["activation"] = "relu"
HP["exact_kl"] = False
HP["pi_0"] = 0.5
HP["pi_1"] = 0.2
HP["pi_2"] = 0.1
HP["soft_0"] = 0.01
HP["adv_eps"] = 0.001
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000
