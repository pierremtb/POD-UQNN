"""Default hyperparameters for 1D time-dep Burgers Equation."""

from podnn.advneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE

HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 1
# Space
HP["n_x"] = 256
HP["x_min"] = 0.
HP["x_max"] = 1.5
# Time
HP["n_t"] = 100
HP["t_min"] = 1.
HP["t_max"] = 5.
# Snapshots count
HP["n_s"] = 50
HP["n_s_hifi"] = int(1e3)
# POD stopping param
HP["eps"] = 0
HP["eps_init"] = None
HP["n_L"] = 20
HP["x_noise"] = 0.01
# Train/val split
HP["train_val_test"] = (30/50, 19/50, 1/50)
# Deep NN hidden layers topology
HP["h_layers"] = [128, 128, 128]
# Setting up TF SGD-based optimizer
HP["n_M"] = 1
HP["epochs"] = 10000
HP["lr"] = 0.01
HP["lambda"] = 0.0
HP["adv_eps"] = 1e+1
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 150
# Burgers params
HP["mu_min"] = [0.001]
HP["mu_max"] = [0.0100]
