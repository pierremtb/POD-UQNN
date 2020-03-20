"""POD-NN modeling for 1D Shekel Equation."""
#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from lib.podnnmodel import PodnnModel
from lib.mesh import create_linear_mesh
from lib.metrics import re_s

#%% Prepare
from hyperparams import HP as hp
from hyperparams import u
print(hp)

resdir = "cache"
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
#%% Model creation
model.initBNN(hp["h_layers"], hp["lr"], 1/X_v_train.shape[0],
              hp["soft_0"], hp["sigma_alea"], hp["norm"])
model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"], freq=hp["log_frequency"])

#%%
v_pred, _ = model.predict_v(X_v_val)
U_pred = model.project_to_U(v_pred)

err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")
