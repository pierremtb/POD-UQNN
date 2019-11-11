""" Run the POD-NN model on 3D steady shallow water equations."""

import os
import sys
import json
import numpy as np

from plots import plot_results

sys.path.append("utils")
from podnn import PodnnModel
from metrics import error_podnn
from mesh import read_space_sol_input_mesh


# HYPER PARAMETERS
if len(sys.argv) > 1:
    with open(sys.argv[1]) as HPFile:
        HP = json.load(HPFile)
else:
    from hyperparams import HP

USE_CACHED_DATASET = True
# USE_CACHED_DATASET = False
USE_TRAINED_NETWORK = True
# USE_TRAINED_NETWORK = False

if not USE_CACHED_DATASET:
    # Getting data from the files
    mu_path = os.path.join("data", "INPUT_100_Scenarios.txt")
    x_u_mesh_path = os.path.join("data", "SOL_FV_100_Scenarios.txt")
    x_mesh, u_mesh, X_v = \
        read_space_sol_input_mesh(HP["n_s"], x_u_mesh_path, mu_path)
    np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
else:
    x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
    u_mesh = None
    X_v = None

# Create the POD-NN model
model = PodnnModel(HP["n_v"], x_mesh, HP["n_t"])

# Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_dataset(u_mesh, X_v,
                                  HP["train_val_ratio"], HP["eps"],
                                  use_cache=USE_CACHED_DATASET, save_cache=True)

# Create the model and train
if not USE_TRAINED_NETWORK:
    def error_val():
        """Defines the error metric for in-training validation."""
        U_val_pred = model.predict(X_v_val)
        return error_podnn(U_val, U_val_pred)
    model.train(X_v_train, v_train, error_val, HP["h_layers"],
                HP["epochs"], HP["lr"], HP["lambda"])
else:
    model.load_trained_cache()

# Predict and restruct
U_pred = model.predict(X_v_val)
U_pred = model.restruct(U_pred)
U_val = model.restruct(U_val)

# Time for one pred
# import time
# st = time.time()
# model.predict(X_v_val[0:1])
# print(f"{time.time() - st} sec taken for prediction")
# exit(0)

# Plot and save the results
plot_results(x_mesh, U_val, U_pred, HP)
