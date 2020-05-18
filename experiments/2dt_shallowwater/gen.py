"""POD-NN modeling of a dam break, with 2D, unsteady Shallow Water Equation."""
#%% Imports
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.mesh import read_multi_space_sol_input_mesh_txt
from poduqnn.handling import clean_dir, split_dataset

from hyperparams import HP as hp

resdir = "cache"
# clean_dir(resdir)

# Setting the dataset split in stone
# fake_x = np.zeros(hp["n_s"] + hp["n_s_tst"])
# test_size = hp["n_s_tst"] / (hp["n_s"] + hp["n_s_tst"])
# train_tst_idx = split_dataset(fake_x, fake_x, test_size, idx_only=True)
# with open(os.path.join("cache", "train_tst_idx.pkl"), "wb") as f:
#     pickle.dump(train_tst_idx, f)
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
    train_tst_idx = pickle.load(f)

# Getting data from the files
datadir = "data"
mu_path = os.path.join(datadir, "INPUT")
sel = np.loadtxt(os.path.join(datadir, "sel.csv"),
                 skiprows=1, delimiter=",")[:, 0].astype("int")
# x_u_mesh_path = datadir
# x_mesh, connectivity, X_v, U, _ = read_multi_space_sol_input_mesh_txt(hp["n_s"], hp["n_t"], hp["d_t"],
#                                                  train_tst_idx[0],
#                                                  hp["mesh_idx"],
#                                                  x_u_mesh_path, mu_path,
#                                                  hp["mu_idx"], sel)
# np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
# np.save(os.path.join("cache", "connectivity.npy"), connectivity)
# np.save(os.path.join("cache", "X_v.npy"), X_v)
# np.save(os.path.join("cache", "U.npy"), U)

x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
connectivity = np.load(os.path.join("cache", "connectivity.npy"))
X_v = np.load(os.path.join("cache", "X_v.npy"))
U = np.load(os.path.join("cache", "U.npy"))

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_multigpu_data(U, X_v, hp["train_val"], hp["eps"], hp["eps_init"])

model.initVNNs(hp["n_M"], hp["h_layers"], hp["lr"], hp["lambda"],
               hp["adv_eps"], hp["soft_0"], hp["norm"])
