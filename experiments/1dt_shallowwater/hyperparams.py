"""Default hyperparameters for 1D time-dep Burgers Equation."""

import numpy as np
from poduqnn.varneuralnetwork import NORM_MEANSTD
from scipy.optimize import brentq

HP = {}
# Dimension of u(x, t, mu)
HP["n_v"] = 2
# Space
HP["n_x"] = 132
HP["x_min"] = 0.
HP["x_max"] = 100.
# Time
HP["n_t"] = 51
HP["t_min"] = 0.
HP["t_max"] = 5.
# Snapshots count
HP["n_s"] = 40
HP["n_s_tst"] = 3
# POD stopping param
HP["eps"] = 1e-3
HP["eps_init"] = 1e-3
HP["n_L"] = 0
HP["x_noise"] = 0.
# Train/val split
HP["train_val"] = (4/5, 1/5)
# Deep NN hidden layers topology
HP["h_layers"] = [256, 256, 256]
# Setting up TF SGD-based optimizer
HP["n_M"] = 5
HP["epochs"] = 50000
HP["lr"] = 0.001
HP["soft_0"] = 0.001
HP["adv_eps"] = 0.0001
HP["lambda"] = 0.0001
HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 500
# Burgers params
HP["mu_min"] = [2.]
HP["mu_max"] = [20.]

def u(X, t, mu, h0=1.):
    """1D Shallow Water analytical solution."""
    """Adapted from https://github.com/python-hydro/pyro2/blob/master/analysis/dam_compare.py."""
    x = X[0]
    h1 = mu[0]
    xmin = x.min()
    xmax = x.max()

    # optimization
    def find_h2(h2):
        return (h2/h1)**3 - 9*(h2/h1)**2*(h0/h1) + \
            16*(h2/h1)**1.5*(h0/h1) - (h2/h1)*(h0/h1)*(h0/h1+8) + \
            (h0/h1)**3
    h2 = brentq(find_h2, min(h0, h1), max(h0, h1))

    # calculate sound speeds
    g = 9.81
    c0 = np.sqrt(g*h0)
    c1 = np.sqrt(g*h1)
    c2 = np.sqrt(g*h2)
    u2 = 2 * (c1 - c2)

    # shock speed
    xi = c0 * np.sqrt(1/8 * ((2*(c2/c0)**2 + 1)**2 - 1))
    xctr = 0.5*(xmin + xmax)

    h_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)
    # h0
    idx = x >= xctr + xi*t
    h_exact[idx] = h0
    u_exact[idx] = 0

    # h1
    idx = x <= xctr - c1*t
    h_exact[idx] = h1
    u_exact[idx] = 0

    # h2
    idx = ((x >= xctr + (u2-c2)*t) & (x < xctr + xi*t))
    h_exact[idx] = h2
    u_exact[idx] = u2

    # h3
    idx = ((x >= xctr - c1*t) & (x < xctr + (u2-c2)*t))
    c3 = 1/3 * (2*c1 - (x-xctr)/t)
    h_exact[idx] = c3[idx]**2 / g
    u_exact[idx] = 2 * (c1-c3[idx])

    return np.vstack((h_exact, u_exact))
