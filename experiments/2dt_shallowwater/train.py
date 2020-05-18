"""POD-NN modeling of a dam break, with 2D, unsteady Shallow Water Equation."""

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
# with open(os.path.join("cache", "train_tst_idx.pkl"), "wb") as f:
#      pickle.dump(train_tst_idx, f)
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
     train_tst_idx = pickle.load(f)

datadir = "data"
mu_path = os.path.join(datadir, "INPUT")
sel = np.loadtxt(os.path.join(datadir, "sel.csv"),
                 skiprows=1, delimiter=",")[:, 0].astype("int")
# x_mesh, connectivity, X_v, U, _ = \
#         read_multi_space_sol_input_mesh_txt(hp["n_s"], hp["n_t"], hp["d_t"],
#                                         train_tst_idx[0],
#                                         hp["mesh_idx"], datadir, mu_path,
#                                         hp["mu_idx"], sel)

# np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
# np.save(os.path.join("cache", "connectivity.npy"), connectivity)
# np.save(os.path.join("cache", "X_v.npy"), X_v)
# np.save(os.path.join("cache", "U.npy"), U)
x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
connectivity = np.load(os.path.join("cache", "connectivity.npy"))
X_v = np.load(os.path.join("cache", "X_v.npy"))
U = np.load(os.path.join("cache", "U.npy"))

#%% Init the model
model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_multigpu_data(U, X_v, hp["train_val"], hp["eps"])
# X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

#%% Model creation
model.initBNN(hp["h_layers"], hp["lr"], 1, hp["activation"],
              hp["exact_kl"],
              pi_0=hp["pi_0"], pi_1=hp["pi_1"], pi_2=hp["pi_2"],
              soft_0=hp["soft_0"], adv_eps=hp["adv_eps"], norm=hp["norm"])
# X_out = np.linspace(500, 1500, 300).reshape(-1, 1)
model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
            freq=hp["log_frequency"], div_max=True)

#%%
#v_pred, v_pred_sig = model.predict_v(X_v_val)
## err_val = re_s(v_val.T, v_pred.T)
## print(f"RE_v: {err_val:4f}")
