"""POD-NN modeling for 1D time-dep Burgers Equation."""
#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.mesh import create_linear_mesh
from poduqnn.handling import pack_layers
from poduqnn.metrics import re_s
from poduqnn.plotting import savefig, figsize
from poduqnn.neupynn import NeuralNetwork

#%% Prepare
from hyperparams import HP as hp
from hyperparams import u
print(hp)

resdir = "cache"
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))
#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, \
    X_v_val, v_val, U_val = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                                    hp["n_s"],
                                                    hp["train_val"],
                                                    hp["eps"], eps_init=hp["eps_init"],
                                                    t_min=hp["t_min"], t_max=hp["t_max"],
                                                    rm_init=True)

# import matplotlib.pyplot as plt
# import meshio
# U_val = model.restruct(U_val)
# mu = int(X_v_val[0, 1])
# vtk = meshio.read(f"data/cas1_{mu}m/0_FV-Paraview_50.vtk")
# sel = np.loadtxt(os.path.join("data", "sel.csv"), skiprows=1, delimiter=",")[:, 6].astype("int")
# x_sim = vtk.points[sel, 1]
# h_sim = vtk.point_data["h"][sel]
# x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
# plt.plot(x, U_val[0, :, 50, 0], "r--")
# plt.plot(x_sim, h_sim, "k--")
# plt.show()
# exit(0)
#%% Train
model.layers = pack_layers(model.n_d, hp["h_layers"], model.n_L)
model.regnn = NeuralNetwork(model.layers, hp["lr"], hp["lambda"], hp["norm"])
# model.initNN(hp["h_layers"], hp["lr"], hp["lambda"], hp["norm"])
model.regnn.fit(X_v_train, v_train, X_v_val, v_val, hp["epochs"])
# model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
#             hp["log_frequency"])

#%% Generate the dataset from the mesh and params
v_pred = model.predict_v(X_v_val)

err_val = re_s(v_val.T, v_pred.T)
print(f"RE_v: {err_val:4f}")
