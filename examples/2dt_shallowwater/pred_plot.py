"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.metrics import re_s
from podnn.plotting import figsize, savefig

from hyperparams import HP as hp

print("Reading paraview results")
U_tst = None
U_pred = None
U_pred_up = None
U_pred_lo = None
x_prime = None
for i in range(hp["n_t"]):
    csv_file = os.path.join("cache", f"x_u_tst_pred.{i}.csv")
    results = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    if i == 0:
        x_prime = results[:, 5]
        U_tst = np.zeros((results.shape[0], hp["n_t"]))
        U_pred = np.zeros_like(U_tst)
        U_pred_up = np.zeros_like(U_tst)
        U_pred_lo = np.zeros_like(U_tst)
    U_tst[:, i] = results[:, 0]
    U_pred[:, i] = results[:, 1]
    U_pred_up[:, i] = results[:, 2]
    U_pred_lo[:, i] = results[:, 3]

    
n_plot_x = 3
n_plot_y = 3
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
idx = [0, 4, hp["n_t"]]
for i, t_i in enumerate(idx):
    ax = fig.add_subplot(gs[i, 1:])
    lbl = r"{\scriptscriptstyle\textrm{tst},1}"
    ax.plot(x_prime, U_pred[:, i], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
    ax.plot(x_prime, U_tst[:, i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
    ax.fill_between(x_prime, U_pred_lo[:, i], U_pred_up[:, i],
                    alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
    ax.set_xlabel(f"$x'\ (t={t_i})$")
    ax.set_ylabel("$\eta$")
    ax.set_title("$s=27.0\ m$")
plt.tight_layout()
savefig("cache/podensnn-swt-samples")

