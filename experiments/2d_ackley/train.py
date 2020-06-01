"""POD-NN modeling for 2D Ackley Equation."""
#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.mesh import create_linear_mesh
from poduqnn.metrics import re_s
from poduqnn.plotting import savefig, figsize
from poduqnn.handling import sample_mu

#%%
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfk = tf.keras
dtype = "float64"
tf.keras.backend.set_floatx(dtype)

#%% Prepare
from hyperparams import HP as hp
from hyperparams import u
print(hp)

resdir = "cache"
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"],
                            hp["y_min"], hp["y_max"], hp["n_y"])
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

#%% Model creation
model.initBNN(hp["h_layers"], hp["lr"], 1, hp["activation"],
              pi_0=hp["pi_0"], pi_1=hp["pi_1"], pi_2=hp["pi_2"],
              soft_0=0.01, soft_1=1e-5,
              norm=hp["norm"])
model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
            freq=hp["log_frequency"])
