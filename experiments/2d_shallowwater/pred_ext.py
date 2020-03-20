"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.mesh import read_space_sol_input_mesh
from poduqnn.plotting import figsize, savefig
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

X_v_in_n_out = np.linspace(500, 1500, 500).reshape(-1, 1)
_, U_pred_sig_in_n_out = model.predict(X_v_in_n_out)
# print(X_v_in_n_out.shape, U_pred_sig_in_n_out.shape)
# print(U_pred_sig_in_n_out.mean(0))
fig = plt.figure(figsize=figsize(1, 1, 2.))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(X_v_in_n_out, U_pred_sig_in_n_out.mean(0), "C0-", label="Prediction uncertainty")
ax.axvline(x=X_v_train.min(), color="k", ls="-.")
ax.axvline(x=X_v_train.max(), color="k", ls="-.", label="Study scope")
print(X_v_train.min())
print(X_v_train.max())
ax.set_xlabel(r"$s=Q$")
ax.set_ylabel(r"$\bar{\sigma}_D(Q)$")
plt.legend()
savefig("in_n_out")
print("saved")
