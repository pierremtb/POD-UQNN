"""POD-NN modeling for 1D, unsteady Burger Equation."""
#%% Imports
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import read_multi_space_sol_input_mesh
from podnn.handling import clean_dir, split_dataset

from hyperparams import HP as hp

resdir = "cache"
clean_dir(resdir)

# Getting data from the files
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
# u_mesh = None
# X_v = None

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.convert_multigpu_data(U, X_v, hp["train_val"], hp["eps"])


model.initVNNs(hp["n_M"], hp["h_layers"],
                hp["lr"], hp["lambda"], hp["adv_eps"], hp["norm"])
