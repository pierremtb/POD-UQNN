"""POD-NN modeling for the 1D time-dep Burgers Equation."""
#%% Imports

import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.plotting import genresultdir

from genhifi import u, generate_test_dataset
from plot import plot_results

#%% Prepare
from hyperparams import HP as hp
print(hp)
resdir = genresultdir()
# generate_test_dataset()

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
                                                    t_min=hp["t_min"], t_max=hp["t_max"],
                                                    x_noise=hp["x_noise"])

#%% Train
model.initVNNs(hp["n_M"], hp["h_layers"],
                hp["lr"], hp["lambda"], hp["adv_eps"], hp["norm"])
train_res = model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
                        freq=hp["log_frequency"])
#%% Predict and restruct
U_val_pred, U_val_pred_sig = model.predict(X_v_val)
U_val_pred = model.restruct(U_val_pred)
U_val_pred_sig = model.restruct(U_val_pred_sig)
U_val = model.restruct(U_val)

#%% Sample the new test set to generate a HiFi prediction
print("Sampling {n_s_hifi} parameters")
X_v_test = model.generate_hifi_inputs(hp["n_s_hifi"],
                                      hp["mu_min"], hp["mu_max"], hp["t_min"], hp["t_max"])
print("Predicting the {n_s_hifi} corresponding solutions")
U_pred_hifi, U_pred_hifi_sig = model.predict(X_v_test)
U_pred_hifi = model.restruct(U_pred_hifi)
U_pred_hifi_sig = model.restruct(U_pred_hifi_sig)
U_pred_hifi_mean = (model.restruct(U_pred_hifi.mean(-1), no_s=True),
                    model.restruct(U_pred_hifi_sig.mean(-1), no_s=True))
U_pred_hifi_std = (model.restruct(U_pred_hifi.std(-1), no_s=True),
                    model.restruct(U_pred_hifi_sig.std(-1), no_s=True))
sigma_pod = model.pod_sig.mean()

# Plot against test and save
plot_results(U_val, U_val_pred, U_pred_hifi_mean, U_pred_hifi_std, sigma_pod,
             resdir, train_res[0], hp)
