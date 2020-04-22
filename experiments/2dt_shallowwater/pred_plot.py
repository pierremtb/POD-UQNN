"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plti

sys.path.append(os.path.join("..", ".."))
from poduqnn.metrics import re_s
from poduqnn.plotting import figsize, savefig

from hyperparams import HP as hp

print("Reading paraview results")
U_tst = None
U_pred = None
U_pred_up = None
U_pred_lo = None
x_prime = None
idx = [0, 5, 50]
for i, t_i in enumerate(idx):
    csv_file = os.path.join("cache", f"x_u_tst_pred.{t_i}.csv")
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

# Load bathymetry
b = np.loadtxt(os.path.join("cache", "b.csv"), delimiter=',', skiprows=1)[:, 5]

    
n_plot_x = 2*len(idx)
n_plot_y = 5
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
ylim = (U_pred_lo[:, 0].min() - 1., U_pred_up[:, 0].max() + 1.)
for i, t_i in enumerate(idx):
    ax = fig.add_subplot(gs[2*i:2*i+2, 0:2])
    img = plti.imread(f"cache/x_u_tst_pred.{t_i}.png")
    ax.imshow(img)
    ax.set_xlabel(r"Surface elevation $\eta$")
    # ax.set_xlabel(f"$x$")
    # ax.set_ylabel(f"$y$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(gs[2*i:2*i+2, 2:])
    lbl = r"{\scriptscriptstyle\textrm{tst},1}"
    # ax.plot(x_prime, b, "k:", label="$b$")
    ax.fill_between(x_prime, np.zeros_like(b), b,
                    edgecolor="k", alpha=0.3, facecolor="w", hatch="/",
                    label="$b$")
    ax.plot(x_prime, U_pred[:, i], "b-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
    ax.plot(x_prime, U_tst[:, i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
    ax.fill_between(x_prime, U_pred_lo[:, i], U_pred_up[:, i],
                    alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
    ax.set_xlabel(f"$x'$")
    ax.set_ylabel("$\eta$")
    ax.set_ylim(ylim)
    ax.set_title(f"$\eta_0=29.368\ m$, $t={t_i * hp['d_t']}\ s$")
    if i == 0:
        ax.legend()
plt.tight_layout()
savefig("results/podensnn-swt-samples", True)

