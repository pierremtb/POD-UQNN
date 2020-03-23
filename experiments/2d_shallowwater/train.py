""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

#%% Import
import sys
import os
import meshio
import pickle
import numpy as np

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.mesh import read_multi_space_sol_input_mesh
from poduqnn.handling import split_dataset
from poduqnn.metrics import re_s, re
from poduqnn.plotting import savefig, figsize

#%% Prepare
from hyperparams import HP as hp
print(hp)

#%% Getting data from the files
# fake_x = np.zeros(hp["n_s"] + hp["n_s_tst"])
# test_size = hp["n_s_tst"] / (hp["n_s"] + hp["n_s_tst"])
# train_tst_idx = split_dataset(fake_x, fake_x, test_size, idx_only=True)
train_tst_idx = ([129, 13, 161, 10, 3, 4, 68, 19, 108, 63, 62, 147, 117, 113, 165, 80, 124, 33, 41, 37, 79, 184, 154, 83, 102, 190, 195, 148, 46, 114, 16, 155, 121, 104, 120, 58, 53, 78, 160, 193, 126, 115, 95, 127, 166, 131, 49, 100, 84, 35, 12, 27, 118, 167, 66, 56, 106, 175, 143, 97, 87, 1, 183, 111, 36, 158, 153, 199, 17, 31, 177, 194, 182, 59, 187, 130, 163, 92, 48, 96, 82, 6, 123, 98, 192, 43, 26, 181, 170, 134, 72, 50, 24, 174, 122, 103, 71, 138, 110, 7, 65, 51, 28, 173, 172, 34, 90, 119, 185, 15, 186, 101, 85, 60, 75, 39, 38, 5, 141, 89, 57, 144, 64, 67, 171, 157, 94, 70, 142, 54, 74, 146, 191, 112, 107, 189, 30, 32, 133, 169, 151, 23, 21, 99, 2, 22, 116, 91, 145, 178, 137, 135, 40, 73, 47, 52, 25, 93, 128, 88, 109, 44, 29, 198, 159, 125, 11, 45, 197, 149, 69, 188, 164, 0, 18, 176, 9, 168, 77, 132], [76, 42, 179, 61, 105, 136, 86, 196, 8, 14, 139, 20, 150, 152, 180, 162, 140, 81, 55, 156])
with open(os.path.join("cache", "train_tst_idx.pkl"), "wb") as f:
     pickle.dump(train_tst_idx, f)

datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_mesh, connectivity, X_v, U = \
        read_multi_space_sol_input_mesh(hp["n_s"], 1, 1, train_tst_idx[0],
                                        hp["mesh_idx"], datadir, mu_path,
                                        hp["mu_idx"])

np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))

#%% Init the model
model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])


#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_multigpu_data(U, X_v, hp["train_val"], hp["eps"])
# X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

#%% Model creation
model.initBNN(hp["h_layers"], hp["lr"], 1/X_v_train.shape[0],
              hp["soft_0"], hp["sigma_alea"], hp["norm"])
X_out = np.linspace(500, 1500, 300).reshape(-1, 1)
model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
            freq=hp["log_frequency"], X_out=X_out)

#%%
v_pred, v_pred_sig = model.predict_v(X_v_val)
# err_val = re_s(v_val.T, v_pred.T)
# print(f"RE_v: {err_val:4f}")

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

U_pred, U_pred_sig = model.predict(X_v_tst)
print(f"RE_tst: {re_s(model.destruct(U_tst), U_pred):4f}")

U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)

#%% VTU export
print("Saving to .vtu")
idx = range(hp["n_s_tst"])
print("Samples are " + ", ".join([f"{X_v_tst[idx[i]].item()}" for i in idx]))
for i, idx_i in enumerate(idx):
    meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_bnn_{i}.vtu"),
                              x_mesh,
                              [("triangle", connectivity)],
                            point_data={
                                "h": U_tst[0, :, idx_i],
                                "h_pred": U_pred[0, :, idx_i],
                                "h_pred_up": U_pred[0, :, idx_i] + 2*U_pred_sig[0, :, idx_i],
                                "h_pred_lo": U_pred[0, :, idx_i] - 2*U_pred_sig[0, :, idx_i],
                                })
