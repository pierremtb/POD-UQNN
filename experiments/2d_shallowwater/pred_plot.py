"""POD-NN modeling for 1D Shekel Equation."""
"""OLD FILE NOT CURRENTLY USED."""
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

# Plotting
csv_file = os.path.join("cache", "x_u_tst_pred.csv")
print("Reading paraview results")
results = np.loadtxt(csv_file, delimiter=',', skiprows=1)
x_line = results[:, 21]
idx = [(8, 9, 10, 11, 12, 13), (14, 15, 16, 17, 18, 19)]
y_axis = ["$(hu)$", "$(hv)$"]
# idx = [(0, 1, 2, 3, 4, 5), (8, 9, 10, 11, 12, 13), (14, 15, 16, 17, 18, 19)]
# y_axis = ["$h$", "$(hu)$", "$(hv)$"]

print("Plotting")
n_plot_x = 2
n_plot_y = 2
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
for col, idx_i in enumerate(idx):
    lbl = r"{\scriptscriptstyle\textrm{tst},1}"
    ax = fig.add_subplot(gs[0, col])
    ax.plot(x_line, results[:, idx[col][1]], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
    ax.plot(x_line, results[:, idx[col][0]], "r--", label=r"$u_D(s_{" + lbl + r"})$")
    lower = results[:, idx[col][1]] - 2*results[:, idx[col][2]]
    upper = results[:, idx[col][1]] + 2*results[:, idx[col][2]]
    ax.fill_between(x_line, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
    ax.set_xlabel("$x'$")
    ax.set_ylabel(y_axis[col])
    if col == len(idx) - 1:
        ax.legend()

    lbl = r"{\scriptscriptstyle\textrm{tst},2}"
    ax = fig.add_subplot(gs[1, col])
    ax.plot(x_line, results[:, idx[col][4]], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
    ax.plot(x_line, results[:, idx[col][3]], "r--", label=r"$u_D(s_{" + lbl + r"})$")
    lower = results[:, idx[col][4]] - 2*results[:, idx[col][5]]
    upper = results[:, idx[col][4]] + 2*results[:, idx[col][5]]
    ax.fill_between(x_line, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
    ax.set_xlabel("$x'$")
    ax.set_ylabel(y_axis[col])
    if col == len(idx) - 1:
        ax.legend()

plt.tight_layout()
# plt.show()
savefig("cache/podnn-sw-graph-samples")
