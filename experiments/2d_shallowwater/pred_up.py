"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import pickle
import meshio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.plotting import figsize, savefig
from poduqnn.mesh import read_multi_space_sol_input_mesh

from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")
# X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

X_v_up = np.linspace(800, 1200, 100).reshape(-1, 1)
U_pred_up, U_pred_sig_up = model.predict(X_v_up)

U_up_mean = U_pred_up.mean(-1)
U_ups_sig = U_pred_up.std(-1)
U_up_var = (U_pred_up**2 + U_pred_sig_up**2).mean(-1) - U_up_mean**2
U_up_sig = np.sqrt(U_up_var)

with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
print(train_tst_idx)
datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = datadir
x_mesh, connectivity, _, _ = \
        read_multi_space_sol_input_mesh(hp["n_s_tst"], 1, 1, train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"])

sel = np.loadtxt(os.path.join(datadir, "selpts.csv"),
                 skiprows=1, delimiter=",")[:, 4].astype("int")
sel_zoom = np.loadtxt(os.path.join(datadir, "selptszoom.csv"),
                 skiprows=1, delimiter=",")[:, 4].astype("int")
x = x_mesh[sel, 0]
y = x_mesh[sel, 1]
x_zoom = x_mesh[sel_zoom, 0]
y_zoom = x_mesh[sel_zoom, 1]

h_0 = 0.05
method = 'linear'
X, Y = np.mgrid[int(x.min()):int(x.max()), int(y.min()):int(y.max())]

h_up = U_up_mean[sel]
h_ups_sig_up = U_up_mean[sel] + 2*U_ups_sig[sel]
h_ups_sig_lo = U_up_mean[sel] - 2*U_ups_sig[sel]
h_up_sig_up = U_up_mean[sel] + 2*U_up_sig[sel]
h_up_sig_lo = U_up_mean[sel] - 2*U_up_sig[sel]

meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_up.vtu"),
                          x_mesh,
                          [("triangle", connectivity)],
                          point_data={
                              "h_up": h_up,
                              "h_ups_sig_up": h_ups_sig_up,
                              "h_ups_sig_lo": h_ups_sig_lo,
                              "h_up_sig_up": h_up_sig_up,
                              "h_up_sig_lo": h_up_sig_lo,
                            })
print("Exported. ParaView processing is now needed to create x_u_tst_pred.csv")

H_up = griddata((x, y), h_up, (X, Y), method=method)
H_ups_sig_up = griddata((x, y), h_ups_sig_up, (X, Y), method=method)
H_ups_sig_lo = griddata((x, y), h_ups_sig_lo, (X, Y), method=method)
H_up_sig_up = griddata((x, y), h_up_sig_up, (X, Y), method=method)
H_up_sig_lo = griddata((x, y), h_up_sig_lo, (X, Y), method=method)

n_plot_x = 1
n_plot_y = 1

fig = plt.figure(figsize=figsize(n_plot_x, n_plot_x, scale=5.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
ax = fig.add_subplot(gs[0, 0])
ax.contour(X, Y, H_up_sig_lo,
        levels=[h_0], alpha=0.3, colors=["C0"])
cs = ax.contour(X, Y, H_up_sig_up,
        levels=[h_0], alpha=0.3, colors=["C0"])
cs.collections[0].set_label(r"$\pm 2\sigma_\textrm{up}(x, y)$")
ax.contour(X, Y, H_ups_sig_lo,
        levels=[h_0], alpha=0.3, colors=["g"])
cs = ax.contour(X, Y, H_ups_sig_up,
        levels=[h_0], alpha=0.3, colors=["g"])
cs.collections[0].set_label(r"$\pm 2\sigma_\textrm{ups}(x, y)$")
cs = ax.contour(X, Y, H_up,
        levels=[h_0], colors=["b"])
cs.collections[0].set_label(r"$\hat{u}_D(x, y)$")
ax.set_aspect('equal', 'box')
ax.set_xlabel(r"$x\ [\textrm{m}]$")
ax.set_ylabel(r"$y\ [\textrm{m}]$")

# ax = fig.add_subplot(gs[0, 1])
# ax.plot(dist_pts[i][0], dist_pts[i][1], "k:")
# dist_i = np.sqrt((dist_pts[i][0][0] - dist_pts[i][0][1])**2 +
#                    (dist_pts[i][1][0] - dist_pts[i][1][1])**2)
# ax.text(dist_pts[i][0][1] - 16, dist_pts[i][1][1] + 9.5,
#         f"$d_\sigma={dist_i:.2f}" + r"\ \textrm{m}$")
# # ax.text(277180, 850+5.048e6, "yo")
# plot_ax(ax)
# ax.legend()
ax.set_xlim((x_zoom.min(), x_zoom.max()))
ax.set_ylim((y_zoom.min(), y_zoom.max()))
plt.tight_layout()
plt.show()
# savefig(os.path.join("results", f"podensnn-sw-zooms-up"))
