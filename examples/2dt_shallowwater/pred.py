"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
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
print(U_pred.shape)
err_val = re_s(U_val[:, 0:1], U_pred[:, 0:1], div_max=True)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
mu_path = os.path.join("..", "..", "..", "scratch", "multi2swt", "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = os.path.join("..", "..", "..", "scratch", "multi2swt")
x_mesh, connectivity_raw, X_v_tst, U_tst = read_multi_space_sol_input_mesh(hp["n_s_tst"], hp["n_t"], hp["d_t"], hp["mesh_idx"],
                                                         x_u_mesh_path, mu_path, hp["mu_idx"],
                                                         n_s_0=0)
U_pred, U_pred_sig = model.predict(X_v_tst)

# print("U_pred", U_pred.shape)
U_pred = model.restruct(U_pred)
U_tst = model.restruct(U_tst)
U_pred_sig = model.restruct(U_pred_sig)
print("U_tst", U_tst.shape)
print("U_pred", U_pred.shape)

# print("U_pred_s", U_pred.shape)

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
idx = np.random.choice(U_pred.shape[-1], 2)
# idx = [50, 250]
# print(f"Samples are {X_v_tst[idx[0]]}, {X_v_tst[idx[1]]}")
for i in range(hp["n_t"]):
    unstructuredGridToVTK(os.path.join("cache", f"x_u_tst_pred_{i}"),
                            x, y, z,
                            connectivity, offsets, cell_types,
                            cellData=None,
                            pointData={
                                # "U_0": np.ascontiguousarray(np.sqrt(U_tst[1, :, idx[0]]**2 + U_tst[2, :, idx[0]]**2)),
                                # "U_1": np.ascontiguousarray(np.sqrt(U_tst[1, :, idx[1]]**2 + U_tst[2, :, idx[1]]**2)),
                                "h_0": np.ascontiguousarray(U_tst[0, :, i, idx[0]]),
                                "h_0_pred": np.ascontiguousarray(U_pred[0, :, i, idx[0]]),
                                "h_0_pred_up": np.ascontiguousarray(U_pred[0, :, i, idx[0]] + 2*U_pred_sig[0, :, i, idx[0]]),
                                "h_0_pred_lo": np.ascontiguousarray(U_pred[0, :, i, idx[0]] - 2*U_pred_sig[0, :, i, idx[0]]),
                                # "h_0_pred_sig": np.ascontiguousarray(U_pred_sig[0, :, idx[0]]),
                                # "h_1": np.ascontiguousarray(U_tst[0, :, idx[1]]),
                                # "h_1_pred": np.ascontiguousarray(U_pred[0, :, idx[1]]),
                                # "h_1_pred_up": np.ascontiguousarray(U_pred[0, :, idx[1]] + 2*U_pred_sig[0, :, idx[1]]),
                                # "h_1_pred_lo": np.ascontiguousarray(U_pred[0, :, idx[1]] - 2*U_pred_sig[0, :, idx[1]]),
                                # "h_1_pred_sig": np.ascontiguousarray(U_pred_sig[0, :, idx[1]]),
                                # "hu_0": np.ascontiguousarray(U_tst[1, :, idx[0]]),
                                # "hu_0_pred": np.ascontiguousarray(U_pred[1, :, idx[0]]),
                                # "hu_0_pred_sig": np.ascontiguousarray(U_pred_sig[1, :, idx[0]]),
                                # "hu_1": np.ascontiguousarray(U_tst[1, :, idx[1]]),
                                # "hu_1_pred": np.ascontiguousarray(U_pred[1, :, idx[1]]),
                                # "hu_1_pred_sig": np.ascontiguousarray(U_pred_sig[1, :, idx[1]]),
                                # "hv_0": np.ascontiguousarray(U_tst[2, :, idx[0]]),
                                # "hv_0_pred": np.ascontiguousarray(U_pred[2, :, idx[0]]),
                                # "hv_0_pred_sig": np.ascontiguousarray(U_pred_sig[2, :, idx[0]]),
                                # "hv_1": np.ascontiguousarray(U_tst[2, :, idx[1]]),
                                # "hv_1_pred": np.ascontiguousarray(U_pred[2, :, idx[1]]),
                                # "hv_1_pred_sig": np.ascontiguousarray(U_pred_sig[2, :, idx[1]]),
                                })
print("Exported. ParaView processing is now needed to create x_u_tst_pred.csv")

