"""Default hyperparameters for 1D time-dep Burgers Equation."""

import numpy as np
from podnn.varneuralnetwork import NORM_MEANSTD, NORM_CENTER, NORM_NONE

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
HP["n_s_tst"] = int(1e3)
# POD stopping param
HP["eps"] = 0
HP["eps_init"] = None
HP["n_L"] = 20
HP["x_noise"] = 0.
# Train/val split
HP["train_val"] = (4/5, 1/5)
# Deep NN hidden layers topology
HP["h_layers"] = [128, 128, 128]
# Setting up TF SGD-based optimizer
HP["n_M"] = 5
HP["epochs"] = 13000
HP["lr"] = 0.01
HP["lambda"] = 0.
HP["adv_eps"] = 0.
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 2000
# Burgers params
HP["mu_min"] = [0.001]
HP["mu_max"] = [0.0100]
HP["mu_min_out"] = [0.0005]
HP["mu_max_out"] = [0.0105]

def u(X, t, mu):
    """Burgers2 explicit solution."""
    x = X[0]
    mu = mu[0]

    if t == 1.:
        res = x / (1 + np.exp(1/(4*mu)*(x**2 - 1/4)))
    else:
        t0 = np.exp(1 / (8*mu))
        res = (x/t) / (1 + np.sqrt(t/t0)*np.exp(x**2/(4*mu*t)))
    return res.reshape((1, x.shape[0]))