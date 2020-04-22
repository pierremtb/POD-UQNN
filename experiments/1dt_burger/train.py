"""POD-NN modeling for 1D time-dep Burgers Equation."""
#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.mesh import create_linear_mesh
from poduqnn.logger import Logger, LoggerCallback
from poduqnn.metrics import re_s
from poduqnn.plotting import savefig, figsize
from poduqnn.handling import sample_mu

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
                                                    t_min=hp["t_min"], t_max=hp["t_max"],
                                                    x_noise=hp["x_noise"])

#%% Train
model.initBNN(hp["h_layers"], hp["lr"], 1,
              hp["activation"],
            #   hp["exact_kl"],
              pi_0=hp["pi_0"], pi_1=hp["pi_1"], pi_2=hp["pi_2"],
              soft_0=0.01, norm=hp["norm"])
model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
            freq=hp["log_frequency"], div_max=False)

#%% Generate the dataset from the mesh and params
v_pred, _ = model.predict_v(X_v_val)
err_val = re_s(v_val.T, v_pred.T)
print(f"RE_v: {err_val:4f}")
