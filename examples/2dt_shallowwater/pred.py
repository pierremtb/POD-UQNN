"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_s
from podnn.mesh import read_multi_space_sol_input_mesh
from podnn.plotting import figsize, savefig
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

#%% Predict and restruct
U_pred, U_pred_sig = model.predict(X_v_val)

#%% Validation metrics
U_pred, _ = model.predict(X_v_val)
err_val = re_s(U_val, U_pred, div_max=True)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
mu_path = os.path.join("..", "..", "..", "scratch", "multi2swt", "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = os.path.join("..", "..", "..", "scratch", "multi2swt")
x_mesh, connectivity_raw, X_v_tst, U_tst = \
        read_multi_space_sol_input_mesh(hp["n_s_tst"], hp["n_t"], hp["d_t"], train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path, hp["mu_idx"])
U_pred, U_pred_sig = model.predict(X_v_tst)

U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)
print(U_tst[...,0])

#%% VTU export
print("Saving to .vtu")
# Retrieving the mesh
# connectivity_raw = np.loadtxt(os.path.join("cache", "connectivity.npy"))
n_element = connectivity_raw.shape[0]

# 1D list of connections
connectivity = connectivity_raw[:, :3].astype("int64").flatten()

# 1d list of "offsets", ie. the end of each element
# Since we use triangles, size = 3
offsets = np.arange(1, n_element + 1) * 3
cell_types = np.ones(n_element, dtype="int64") * VtkTriangle.tid

# Space points
x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
x = x_mesh[:, 0]
y = x_mesh[:, 1]
x = np.ascontiguousarray(x)
y = np.ascontiguousarray(y)
z = np.ascontiguousarray(np.zeros_like(x))

# Exporting
idx = [0]
print(train_tst_idx[1])
print(f"Sample is {X_v_tst[idx[0]*hp['n_t']][1]}")
for i in range(hp["n_t"]):
    unstructuredGridToVTK(os.path.join("cache", f"x_u_tst_pred_{i}"),
                            x, y, z,
                            connectivity, offsets, cell_types,
                            cellData=None,
                            pointData={
                                "eta_0": np.ascontiguousarray(U_tst[0, :, i, idx[0]]),
                                "eta_0_pred": np.ascontiguousarray(U_pred[0, :, i, idx[0]]),
                                "eta_0_pred_up": np.ascontiguousarray(U_pred[0, :, i, idx[0]] + 2*U_pred_sig[0, :, i, idx[0]]),
                                "eta_0_pred_lo": np.ascontiguousarray(U_pred[0, :, i, idx[0]] - 2*U_pred_sig[0, :, i, idx[0]]),
                                })
print("Exported. ParaView processing is now needed to create x_u_tst_pred.csv")

