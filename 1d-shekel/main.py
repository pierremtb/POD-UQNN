import sys
import numpy as np
import tensorflow as tf

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
sys.path.append("utils")
from pod import get_pod_bases, prep_data
from nn import NeuralNetwork
from logger import Logger

# HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Space (dx = 1/30, n_e = 10/dx)
    hp["n_e"] = 300
    # Snapshots count
    hp["n_t"] = 100
    # POD stopping param
    hp["eps"] = 1e-10
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 10000
    hp["tf_lr"] = 0.001
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    hp["log_frequency"] = 100

# Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
# u_rb are the reduced coefficients we're looking for
x, S, X_u_rb_train = prep_data(hp["n_e"], hp["n_t"])
V = get_pod_bases(S, hp["n_e"], hp["n_t"], hp["eps"])
n_L = V.shape[1]

# Projecting
u_rb_train = (V.T.dot(S)).T

# Neural net
# In: (x, mu_1, ..., mu_n)
# Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
hp["layers"] = [3, 10, 20, n_L]
logger = Logger(hp)
def error():
    return 0.0
logger.set_error_fn(error)
model = NeuralNetwork(hp, logger)

import matplotlib.pyplot as plt

for i in range(2):
    plt.plot(np.sort(X_u_rb_train[:, 0]), u_rb_train[:, i][np.argsort(X_u_rb_train[:, 0])])
plt.show()

model.fit(X_u_rb_train, u_rb_train)
