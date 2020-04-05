"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import pickle
import meshio
import numpy as np
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

#%% Sample the new model to generate a test prediction
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
print(train_tst_idx)
datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = datadir
x_mesh, connectivity, X_v_tst, U_tst = \
        read_multi_space_sol_input_mesh(hp["n_s_tst"], 1, 1, train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"])

U_pred, U_pred_sig = model.predict(X_v_tst)
print(f"RE_tst: {re_s(model.destruct(U_tst), U_pred):4f}")

U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)

#%%
sel = np.loadtxt(os.path.join(datadir, "selpts.csv"),
                 skiprows=1, delimiter=",")[:, 4].astype("int")
sel_zoom = np.loadtxt(os.path.join(datadir, "selptszoom.csv"),
                 skiprows=1, delimiter=",")[:, 4].astype("int")

#%%
idx = range(hp["n_s_tst"])

x = x_mesh[sel, 0]
y = x_mesh[sel, 1]
x_zoom = x_mesh[sel_zoom, 0]
y_zoom = x_mesh[sel_zoom, 1]

h_0 = 0.05
method = 'linear'
X, Y = np.mgrid[int(x.min()):int(x.max()), int(y.min()):int(y.max())]

dist_pts = [([277183, 277179.55], [5048840, 5048842.38]),
            ([277212.77, 277205.61], [5048832.65, 5048838.01])]

for i, s in enumerate([idx[2], idx[4]]):
        print(s)
        h = U_tst[0, sel, s]
        h_pred = U_pred[0, sel, s]
        h_pred_up = U_pred[0, sel, s] + 2 * U_pred_sig[0, sel, s]
        h_pred_lo = U_pred[0, sel, s] - 2 * U_pred_sig[0, sel, s]

        H_pred = griddata((x, y), h_pred, (X, Y), method=method)
        H_pred_up = griddata((x, y), h_pred_up, (X, Y), method=method)
        H_pred_lo = griddata((x, y), h_pred_lo, (X, Y), method=method)
        H = griddata((x, y), h, (X, Y), method=method)
        n_plot_x = 1
        n_plot_y = 2

        def plot_ax(ax):
                ax.contour(X, Y, H_pred_lo,
                        levels=[h_0], alpha=0.3, colors=["C0"])
                cs = ax.contour(X, Y, H_pred_up,
                levels=[h_0], alpha=0.3, colors=["C0"])
                cs.collections[0].set_label(r"$\pm 2\hat{u}^\sigma_D$")
                cs = ax.contour(X, Y, H_pred,
                        levels=[h_0], colors=["b"])
                cs.collections[0].set_label(r"$\hat{u}^\mu_D$")
                cs = ax.contour(X, Y, H,
                        levels=[h_0], colors=["r"], linestyles="dashed")
                cs.collections[0].set_label(r"$u_D$")
                ax.set_aspect('equal', 'box')
                ax.set_xlabel(r"$x\ [\textrm{m}]$")
                ax.set_ylabel(r"$y\ [\textrm{m}]$")

        fig = plt.figure(figsize=figsize(n_plot_x, n_plot_x, scale=5.0))
        gs = fig.add_gridspec(n_plot_x, n_plot_y)
        ax = fig.add_subplot(gs[0, 0])
        plot_ax(ax)
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(dist_pts[i][0], dist_pts[i][1], "k:", label="$d_{2\sigma}$")
        dist_i = np.sqrt((dist_pts[i][0][0] - dist_pts[i][0][1])**2 +
                           (dist_pts[i][1][0] - dist_pts[i][1][1])**2)
        ax.text(dist_pts[i][0][1] - 16, dist_pts[i][1][1] + 4.5,
                f"$d_\sigma={dist_i:.2f}" + r"\ \textrm{m}$")
        # ax.text(277180, 850+5.048e6, "yo")
        plot_ax(ax)
        ax.legend()
        ax.set_xlim((x_zoom.min(), x_zoom.max()))
        ax.set_ylim((y_zoom.min(), y_zoom.max()))
        plt.tight_layout()
        # plt.show()
        savefig(os.path.join("results", f"podbnn-sw-zooms-{s}"))

#%% VTU export
print("Saving to .vtu")
print("Samples are " + ", ".join([f"{X_v_tst[idx[i]].item()}" for i in idx]))
for i, idx_i in enumerate(idx):
    meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_bnn_{i}.vtu"),
                              x_mesh,
                              [("triangle", connectivity)],
                              point_data={
                                  "h": U_tst[0, :, idx_i],
                                  "h_pred": U_pred[0, :, idx_i],
                                  "h_pred_up": U_pred[0, :, idx_i] + 2*U_pred_sig[0, :, idx_i],
                                  "h_pred_lo": U_pred[0, :, idx_i] - 2*U_pred_sig[0, :, idx_i],
                                })
print("Exported. ParaView processing is now needed to create x_u_tst_pred.csv")
