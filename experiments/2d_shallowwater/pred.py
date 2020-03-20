"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import pickle
import meshio
import numpy as np

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.mesh import read_multi_space_sol_input_mesh
from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

v_pred_mean, sig_alea = model.predict_v(X_v_train)
_, sig_alea_val = model.predict_v(X_v_val)
print(sig_alea.mean(), sig_alea.min(), sig_alea.max())
print(sig_alea_val.mean(), sig_alea_val.min(), sig_alea_val.max())
pod_sig_v = np.stack((v_train, v_pred_mean), axis=-1).std(-1).mean(0)
print(pod_sig_v.mean(), pod_sig_v.min(), pod_sig_v.max())

#%% Predict and restruct
U_pred, U_pred_sig = model.predict(X_v_val)

#%% Validation metrics
U_pred, _ = model.predict(X_v_val)
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
print(train_tst_idx)
# datadir = os.path.join("..", "..", "..", "scratch", "multi2swt") 
datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = datadir
x_mesh, connectivity, X_v_tst, U_tst = \
        read_multi_space_sol_input_mesh(hp["n_s_tst"], 1, 1, train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"])

U_pred, U_pred_sig = model.predict(X_v_tst)
print(f"RE_tst: {re_s(model.destruct(U_tst), U_pred):4f}")

U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)

#%% VTU export
print("Saving to .vtu")
idx = range(hp["n_s_tst"])
print("Samples are " + ", ".join([f"{X_v_tst[idx[i]].item()}" for i in idx]))
for i, idx_i in enumerate(idx):
    meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_{i}.vtu"),
                              x_mesh,
                              [("triangle", connectivity)],
                              point_data={
                                  "h": U_tst[0, :, idx_i],
                                  "h_pred": U_pred[0, :, idx_i],
                                  "h_pred_up": U_pred[0, :, idx_i] + 2*U_pred_sig[0, :, idx_i],
                                  "h_pred_lo": U_pred[0, :, idx_i] - 2*U_pred_sig[0, :, idx_i],
                                })
print("Exported. ParaView processing is now needed to create x_u_tst_pred.csv")
