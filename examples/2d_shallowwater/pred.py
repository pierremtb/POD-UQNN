""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import read_space_sol_input_mesh 
from podnn.metrics import re_s, re
from podnn.plotting import savefig, figsize
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

#%% Prepare
from hyperparams import HP as hp

#%% Getting data from the files
# mu_path = os.path.join("data", f"INPUT_{hp['n_s']}_Scenarios.txt")
# x_u_mesh_path = os.path.join("data", f"SOL_FV_{hp['n_s']}_Scenarios.txt")
# x_mesh, u_mesh, X_v = \
#     read_space_sol_input_mesh(hp["n_s"], hp["mesh_idx"], x_u_mesh_path, mu_path)
# np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
# u_mesh = None
# X_v = None

#%% Init the model
model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])


mu_path_tst = os.path.join("data", f"INPUT_{hp['n_s_tst']}_Scenarios.txt")
x_u_mesh_tst_path = os.path.join("data", f"SOL_FV_{hp['n_s_tst']}_Scenarios.txt")
x_mesh, u_mesh_tst, X_v_tst = \
    read_space_sol_input_mesh(hp["n_s_tst"], hp["mesh_idx"], x_u_mesh_tst_path, mu_path_tst)
U_tst = model.u_mesh_to_U(u_mesh_tst, hp["n_s_tst"])
print(f"RE_tst: {re(U_tst, U_tst):4f}")