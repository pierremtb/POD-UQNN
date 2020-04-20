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

from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")

X_v_in_n_out = np.linspace(500, 1500, 300).reshape(-1, 1)
_, U_pred_sig_in_n_out = model.predict(X_v_in_n_out)
fig = plt.figure(figsize=figsize(1, 1, 2.))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.plot(X_v_in_n_out, U_pred_sig_in_n_out.mean(0), "b-", label="Prediction uncertainty")
ax.axvline(x=800, color="k", ls="-.")
ax.axvline(x=1200, color="k", ls="-.", label="Study scope")
ax.set_xlabel(r"$s=Q\ [\textrm{m}/\textrm{s}]$")
ax.set_ylabel(r"$\bar{\sigma}_h\ [\textrm{m}]$")
ax.set_ylim((0, 0.5))
plt.legend()
savefig(os.path.join("results", "in_n_out"))
