""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

#%% Import
import sys
import os
import meshio
import pickle
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import read_multi_space_sol_input_mesh
from podnn.handling import split_dataset
from podnn.metrics import re_s, re
from podnn.plotting import savefig, figsize

#%% Prepare
from hyperparams import HP as hp
print(hp)

#%% Getting data from the files
fake_x = np.zeros(hp["n_s"] + hp["n_s_tst"])
test_size = hp["n_s_tst"] / (hp["n_s"] + hp["n_s_tst"])
train_tst_idx = split_dataset(fake_x, fake_x, test_size, idx_only=True)
with open(os.path.join("cache", "train_tst_idx.pkl"), "wb") as f:
     pickle.dump(train_tst_idx, f)

datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_mesh, connectivity, X_v, U = \
        read_multi_space_sol_input_mesh(hp["n_s"], 1, 1, train_tst_idx[0],
                                        hp["mesh_idx"], datadir, mu_path,
                                        hp["mu_idx"])

np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)

#%% Init the model
model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])


#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_multigpu_data(U, X_v, hp["train_val"], hp["eps"])

#%% Model creation
model.initBNN(hp["h_layers"], hp["lr"], 1/X_v_train.shape[0],
              hp["soft_0"], hp["sigma_alea"], hp["norm"])
model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
            freq=hp["log_frequency"])

#%%
v_pred, v_pred_sig = model.predict_v(X_v_val)
# err_val = re_s(v_val.T, v_pred.T)
# print(f"RE_v: {err_val:4f}")

#%%
# import matplotlib.pyplot as plt
# yhat = model.regnn.predict_dist(X_v_val)
# for i in [0, 1]:
#     plt.plot(yhat.mean().numpy()[i], "b-")
#     plt.plot(yhat.mean().numpy()[i] - 2*yhat.stddev().numpy()[i], "b-", alpha=0.2)
#     plt.plot(yhat.mean().numpy()[i] + 2*yhat.stddev().numpy()[i], "b-", alpha=0.2)
#     plt.plot(v_val[i], "r--")
# plt.show()

# y_pred, y_pred_sig = model.predict_v(X_v_val)
# for i in [0, 1]:
#     plt.plot(y_pred[i], "b-")
#     plt.plot(y_pred[i] - 2*y_pred_sig[i], "b-", alpha=0.2)
#     plt.plot(y_pred[i] + 2*y_pred_sig[i], "b-", alpha=0.2)
#     plt.plot(v_val[i], "r--")
# plt.show()

#%% Cleanup
# del x_mesh, u_mesh, X_v, X_v_train, v_train, X_v_val, v_val, U_val, v_pred, v_pred_sig

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
    meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_bnn_{i}.vtu"),
                              x_mesh,
                              [("triangle", connectivity)],
                            point_data={
                                "h": U_tst[0, :, idx_i],
                                "h_pred": U_pred[0, :, idx_i],
                                "h_pred_up": U_pred[0, :, idx_i] + 2*U_pred_sig[0, :, idx_i],
                                "h_pred_lo": U_pred[0, :, idx_i] - 2*U_pred_sig[0, :, idx_i],
                                # "U": np.ascontiguousarray(np.sqrt(U_tst[1, :, idx[i]]**2 + U_tst[2, :, idx[i]]**2)),
                                # "h_pred_sig": np.ascontiguousarray(U_pred_sig[0, :, idx[i]]),
                                # "hu": np.ascontiguousarray(U_tst[1, :, idx[i]]),
                                # "hu_pred": np.ascontiguousarray(U_pred[1, :, idx[i]]),
                                # "hu_pred_sig": np.ascontiguousarray(U_pred_sig[1, :, idx[i]]),
                                # "hv": np.ascontiguousarray(U_tst[2, :, idx[i]]),
                                # "hv_pred": np.ascontiguousarray(U_pred[2, :, idx[i]]),
                                # "hv_pred_sig": np.ascontiguousarray(U_pred_sig[2, :, idx[i]]),
                                })


# %%
