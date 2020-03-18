"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import pickle
import meshio
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
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
# datadir = os.path.join("..", "..", "..", "scratch", "multi2swt") 
datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = datadir
x_mesh, connectivity, X_v_tst, U_tst = \
        read_multi_space_sol_input_mesh(hp["n_s_tst"], 1, 1, train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"])
U_tst = model.destruct(U_tst)
U_pred, U_pred_sig = model.predict(X_v_tst)

print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

U_tst = model.restruct(U_tst)
U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)

#%% VTU export
print("Saving to .vtu")
idx = [0, 1]
print(f"Samples are {X_v_tst[idx[0]]}, {X_v_tst[idx[1]]}")
for i, idx_i in enumerate(idx):
    meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_{i}.vtu"),
                            x_mesh,
                            [("triangle", connectivity)],
                            point_data={
                                "h": U_tst[0, :, idx_i],
                                "h_pred": U_pred[0, :, idx_i],
                                "h_pred_up": U_pred[0, :, idx_i] + 2*U_pred_sig[0, :, idx_i],
                                "h_pred_lo": U_pred[0, :, idx_i] - 2*U_pred_sig[0, :, idx_i],
                                # "U": np.ascontiguousarray(np.sqrt(U_tst[1, :, idx[0]]**2 + U_tst[2, :, idx[0]]**2)),
                                # "h_pred_sig": np.ascontiguousarray(U_pred_sig[0, :, idx[0]]),
                                # "hu": np.ascontiguousarray(U_tst[1, :, idx[0]]),
                                # "hu_pred": np.ascontiguousarray(U_pred[1, :, idx[0]]),
                                # "hu_pred_sig": np.ascontiguousarray(U_pred_sig[1, :, idx[0]]),
                                # "hv": np.ascontiguousarray(U_tst[2, :, idx[0]]),
                                # "hv_pred": np.ascontiguousarray(U_pred[2, :, idx[0]]),
                                # "hv_pred_sig": np.ascontiguousarray(U_pred_sig[2, :, idx[0]]),
                                })
print("Exported. ParaView processing is now needed to create x_u_tst_pred.csv")

