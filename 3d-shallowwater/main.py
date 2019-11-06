import os
import sys
import json
import numpy as np
import tensorflow as tf

eqnPath = "3d-shallowwater"
sys.path.append(eqnPath)
from plots import plot_results

sys.path.append("utils")
from podnn import PodnnModel
from metrics import error_podnn
from mesh import read_space_sol_input_mesh


# HYPER PARAMETERS
if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    from hyperparams import hp

use_cache = True
# use_cache = False

if not use_cache:
    # Getting data from the files
    mu_path = os.path.join(eqnPath, "data", "INPUT_100_Scenarios.txt")
    x_u_mesh_path = os.path.join(eqnPath, "data", "SOL_FV_100_Scenarios.txt")
    x_mesh, u_mesh, X_v = \
        read_space_sol_input_mesh(hp["n_s"], x_u_mesh_path, mu_path)
    np.save(os.path.join(eqnPath, "cache", "x_mesh.npy"), x_mesh)
else:
    x_mesh = np.load(os.path.join(eqnPath, "cache", "x_mesh.npy"))
    u_mesh = None
    X_v = None
    
# Extend the class and init the model
model = PodnnModel(hp["n_v"], x_mesh, hp["n_t"], eqnPath)

# Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_dataset(u_mesh, X_v,
                                  hp["train_val_ratio"], hp["eps"],
                                  use_cache=use_cache, save_cache=True)


# Train
def error_val():
    U_pred = model.predict(X_v_val)
    return error_podnn(U_val, U_pred)
model.train(X_v_train, v_train, error_val, hp["h_layers"],
            hp["epochs"], hp["lr"], hp["lambda"]) 

# Predict and restruct
U_pred = model.predict(X_v_val)

U_pred = model.restruct(U_pred)
U_val = model.restruct(U_val)

# # PLOTTING AND SAVING RESULTS
plot_results(x_mesh, U_val, U_pred, hp, eqnPath)
