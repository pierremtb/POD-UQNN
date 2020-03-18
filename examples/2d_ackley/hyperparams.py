"""Default hyperparameters for 2D Ackley Equation."""

import numpy as np
import tensorflow as tf

from podnn.bayesianneuralnetwork import NORM_MINMAX, NORM_MEANSTD


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
HP["h_layers"] = [40, 40]
# Setting up TF SGD-based optimizer
HP["epochs"] = 45000
HP["lr"] = 0.005
HP["soft_0"] = 0.05
HP["sigma_alea"] = 2.76
HP["norm"] = NORM_MINMAX
# HP["norm"] = NORM_MEANSTD
# Frequency of the logger
HP["log_frequency"] = 1000
# Non-spatial params
HP["mu_min"] = [-1., -1., -1.]
HP["mu_max"] = [+1., +1., +1.]
HP["mu_min_out"] = [-2., -2., -2.]
HP["mu_max_out"] = [+2., +2., +2.]


# np.random.seed(1111)
# tf.random.set_seed(1111)

def u(X, _, mu):
    x = X[0]
    y = X[1]
    u_0 = - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
          - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
          + 20 + np.exp(1)
    return u_0.reshape((1, u_0.shape[0]))
