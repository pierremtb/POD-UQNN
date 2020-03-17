"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_s
from podnn.mesh import read_space_sol_input_mesh
from podnn.plotting import figsize, savefig
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, U_val = model.load_train_data()


mu_path_tst = os.path.join("data", f"INPUT_{hp['n_s_tst']}_Scenarios.txt")
x_u_mesh_tst_path = os.path.join("data", f"SOL_FV_{hp['n_s_tst']}_Scenarios.txt")
x_mesh, u_mesh_tst, X_v_tst = \
    read_space_sol_input_mesh(hp["n_s_tst"], hp["mesh_idx"], x_u_mesh_tst_path, mu_path_tst)
U_tst = model.u_mesh_to_U(u_mesh_tst, hp["n_s_tst"])
print(f"RE_tst: {re(U_tst, U_tst):4f}")