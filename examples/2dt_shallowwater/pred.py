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
err_val = re_s(U_val, U_pred, div_max=True)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
# datadir = os.path.join("..", "..", "..", "scratch", "multi2swt") 
datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = datadir
sel = np.loadtxt('sel.txt', skiprows=6)[:, 1].astype("int")
x_mesh, connectivity_raw, X_v_tst, U_tst = \
        read_multi_space_sol_input_mesh(hp["n_s_tst"], hp["n_t"], hp["d_t"], train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"], sel)
# U_pred, U_pred_sig = model.predict(X_v_tst)
v_pred, v_pred_sig = model.predict_v(X_v_tst)
U_pred = model.project_to_U(v_pred)
U_pred_sig = model.project_to_U(v_pred_sig)
U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)

#%% VTU export
print("Saving to .vtu")
idx = [0, 1, 2]

connectivity = connectivity_raw[:, :3].astype("int64")
for s in range(len(idx)):
    print(f"Sample is {X_v_tst[idx[s]*hp['n_t']][1]}")
    for i in range(hp["n_t"]):
        meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_{s}.{i}.vtu"),
                                x_mesh,
                                [("triangle", connectivity)],
                                point_data={
                                        "eta_0": U_tst[0, :, i, idx[s]],
                                        "eta_0_pred": U_pred[0, :, i, idx[s]],
                                        "eta_0_pred_up": U_pred[0, :, i, idx[s]] + 2*U_pred_sig[0, :, i, idx[s]],
                                        "eta_0_pred_lo": U_pred[0, :, i, idx[s]] - 2*U_pred_sig[0, :, i, idx[s]],
                                        })


# %%
