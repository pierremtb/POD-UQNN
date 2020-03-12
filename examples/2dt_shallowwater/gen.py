"""POD-NN modeling for 1D, unsteady Burger Equation."""
#%% Imports
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import read_multi_space_sol_input_mesh
from podnn.handling import clean_dir

from hyperparams import HP as hp

resdir = "cache"
clean_dir(resdir)

# Getting data from the files
# mu_path = os.path.join("data", f"INPUT_{hp['n_s']}_Scenarios.txt")
# x_u_mesh_path = os.path.join("data", f"SOL_FV_{hp['n_s']}_Scenarios.txt")
mu_path = os.path.join("..", "..", "..", "scratch", "multi2swt", "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = os.path.join("..", "..", "..", "scratch", "multi2swt")
x_mesh, connectivity, U, X_v = read_multi_space_sol_input_mesh(hp["n_s"], hp["n_t"], hp["d_t"], hp["mesh_idx"],
                                                 x_u_mesh_path, mu_path, hp["mu_idx"])
np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
np.save(os.path.join("cache", "connectivity.npy"), connectivity)
# x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
# u_mesh = None
# X_v = None

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_multigpu_data(U, X_v, hp["train_val"], hp["eps"])

model.initVNNs(hp["n_M"], hp["h_layers"],
                hp["lr"], hp["lambda"], hp["adv_eps"], hp["norm"])
