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
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

X_v_in_n_out = np.linspace(700, 1300, 500).reshape(-1, 1)
_, U_pred_sig_in_n_out = model.predict(X_v_in_n_out)
# print(X_v_in_n_out.shape, U_pred_sig_in_n_out.shape)
# print(U_pred_sig_in_n_out.mean(0))
fig = figsize(1, 1, size=2.)
plt.plot(X_v_in_n_out, U_pred_sig_in_n_out.mean(0), "b-", "Prediction uncertainty")
plt.axvline(x=950, color="k", ls="-.")
plt.axvline(x=1150, color="k", ls="-.", label="Study scope")
plt.xlabel(r"$s=Q$")
plt.ylabel(r"$\bar{\sigma}_D(Q)$")
savefig("in_n_out.pdf")

