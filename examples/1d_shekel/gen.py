"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.plotting import genresultdir, figsize, savefig
from podnn.metrics import re_s

resdir = genresultdir()

def u(X, _, mu):
    """The 1D-Shekel function."""
    x = X[0]
    sep = int(mu.shape[0] / 2)
    bet = mu[:sep]
    gam = mu[sep:]

    u_sum = np.zeros_like(x)
    for i in range(len(bet)):
        i_sum = (x - gam[i])**2
        u_sum += 1 / (bet[i] + i_sum)

    return u_sum.reshape((1, u_sum.shape[0]))

# Create linear space mesh
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, _, \
    X_v_val, v_val, U_val = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                                    hp["n_s"],
                                                    hp["train_val"],
                                                    eps=hp["eps"], n_L=hp["n_L"],
                                                    u_noise=hp["u_noise"],
                                                    x_noise=hp["x_noise"])

model.initVNNs(hp["n_M"], hp["h_layers"],
                hp["lr"], hp["lambda"], hp["adv_eps"], hp["norm"])