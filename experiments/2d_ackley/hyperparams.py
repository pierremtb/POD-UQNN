"""Default hyperparameters for 2D Ackley Equation."""

import numpy as np

from poduqnn.custombnn import NORM_MINMAX


HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 1
# Space
HP["n_x"] = 400
HP["x_min"] = -5
HP["x_max"] = +5.
HP["n_y"] = 400
HP["y_min"] = -5.
HP["y_max"] = +5.
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
HP["h_layers"] = [140, 140]
# Setting up TF SGD-based optimizer
HP["epochs"] = 15000
HP["lr"] = 0.003
HP["soft_0"] = 0.01
HP["sigma_alea"] = 0.1
HP["norm"] = NORM_MINMAX
# Frequency of the logger
HP["log_frequency"] = 1000
# Non-spatial params
HP["mu_min"] = [-1., -1., -1.]
HP["mu_max"] = [+1., +1., +1.]
HP["mu_min_out"] = [-2., -2., -2.]
HP["mu_max_out"] = [+2., +2., +2.]


def u(X, _, mu):
    """The stochastic 2D Ackley Function."""
    x = X[0]
    y = X[1]
    u_0 = - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
          - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
          + 20 + np.exp(1)
    return u_0.reshape((1, u_0.shape[0]))
