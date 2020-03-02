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

# Plotting
# csv_file = os.path.join("cache", "x_u_tst_pred.csv")
# print("Reading paraview results")
# results = np.loadtxt(csv_file, delimiter=',', skiprows=1)
# x_line = results[:, 21]
# idx = [(8, 9, 10, 11, 12, 13), (14, 15, 16, 17, 18, 19)]
# y_axis = ["$(hu)$", "$(hv)$"]
print("Reading paraview results")
U_tst_list = []
x_prime = None
idx = [0, 4]
for i, file in enumerate(idx):
    csv_file = os.path.join("cache", f"multi2swt_face_{file}.csv")
    results = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    print(results.shape)
    if i == 0:
        x_prime = results[:, 7]
        U_tst = np.zeros((results.shape[0], len(idx)))
    U_tst[:, i] = results[:, 4]

U_pred = U_tst

    
n_plot_x = 2
n_plot_y = 2
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
for i in range(len(idx)):
    ax = fig.add_subplot(gs[i, :])
    ax.plot(x_prime, U_pred[:, i], "b-")
    ax.plot(x_prime, U_tst[:, i], "r--")
    if i == 0:
        ax.set_xlabel("$x'\ (t=0)$")
    else:
        ax.set_xlabel("$x'\ (t=4\ s)$")
    ax.set_ylabel("$\eta$")
    ax.set_title("$s=27.0\ m$")
plt.tight_layout()
plt.savefig("cache/podensnn-swt-samples.pdf")
# plt.show()

print(U_tst)
exit(0)

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
