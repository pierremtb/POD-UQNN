""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_s
from podnn.mesh import read_space_sol_input_mesh
from podnn.plotting import figsize, savefig
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

#%% Prepare
from hyperparams import HP as hp

# Getting data from the files
mu_path = os.path.join("data", f"INPUT_{hp['n_s']}_Scenarios.txt")
x_u_mesh_path = os.path.join("data", f"SOL_FV_{hp['n_s']}_Scenarios.txt")
x_mesh, u_mesh, X_v = \
    read_space_sol_input_mesh(hp["n_s"], hp["mesh_idx"], x_u_mesh_path, mu_path)
np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
# u_mesh = None
# X_v = None

#%% Init the model
model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_dataset(u_mesh, X_v,
                                    hp["train_val"], hp["eps"])

#%% Train
model.initNN(hp["h_layers"], hp["lr"], hp["lambda"])
train_res = model.train(X_v_train, v_train, X_v_train, v_train, hp["epochs"],
                        freq=hp["log_frequency"])

#%% Validation metrics
U_pred = model.predict(X_v_val)
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
mu_path_tst = os.path.join("data", f"INPUT_{hp['n_s_tst']}_Scenarios.txt")
x_u_mesh_tst_path = os.path.join("data", f"SOL_FV_{hp['n_s_tst']}_Scenarios.txt")
_, u_mesh_tst, X_v_tst = \
    read_space_sol_input_mesh(hp["n_s_tst"], hp["mesh_idx"], x_u_mesh_tst_path, mu_path_tst)
U_tst = model.u_mesh_to_U(u_mesh_tst, hp["n_s_tst"])
U_pred = model.predict(X_v_tst)

print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

#%% VTU export
print("Saving to .vtu")
# Retrieving the mesh
connectivity_raw = np.loadtxt(os.path.join("data", "connectivity.txt"))
n_element = connectivity_raw.shape[0]

# 1D list of connections
connectivity = connectivity_raw[:, 1:4].astype("int64").flatten() - 1

# 1d list of "offsets", ie. the end of each element
# Since we use triangles, size = 3
offsets = np.arange(1, n_element + 1) * 3
cell_types = np.ones(n_element, dtype="int64") * VtkTriangle.tid

# Space points
x = x_mesh[:, 1]
y = x_mesh[:, 2]
x = np.ascontiguousarray(x)
y = np.ascontiguousarray(y)
z = np.ascontiguousarray(np.zeros_like(x))

# Exporting
unstructuredGridToVTK(os.path.join("cache", "x_u_test_pred_mean_std"),
                        x, y, z,
                        connectivity, offsets, cell_types,
                        cellData=None,
                        pointData={
                            "h_mean" : U_tst.mean(-1)[0],
                            "h_pred_mean" : U_pred.mean(-1)[0],
                            })
print("Exported. ParaView processing is now needed to create x_u_pred_mean_std.csv")
# plot_results(os.path.join("cache", "x_u_pred_mean_std.csv"), hp)
