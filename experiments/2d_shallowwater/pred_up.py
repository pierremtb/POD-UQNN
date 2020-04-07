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

X_v_up = np.linspace(800, 1200, 400).reshape(-1, 1)
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
        read_multi_space_sol_input_mesh(hp["n_s_tst"], 1, 1, [0],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"])

meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_up.vtu"),
                          x_mesh,
                          [("triangle", connectivity)],
                          point_data={
                              "h_up": U_up_mean,
                              "h_ups_sig_up": U_up_mean + 2*U_ups_sig,
                              "h_ups_sig_lo": U_up_mean - 2*U_ups_sig,
                              "h_up_sig_up": U_up_mean + 2*U_up_sig,
                              "h_up_sig_lo": U_up_mean - 2*U_up_sig,
                            })
print("Exported. ParaView processing is now needed to create x_u_tst_pred.csv")

sel1 = np.loadtxt(os.path.join(datadir, "selptszoom.csv"),
                 skiprows=1, delimiter=",")[:, 4].astype("int")
sel2 = np.loadtxt(os.path.join(datadir, "selpts2.csv"),
                 skiprows=1, delimiter=",")[:, 5].astype("int")

h_0 = 0.05
method = 'linear'

n_plot_x = 1
n_plot_y = 2
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_x, scale=5.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)

dist_pts1 = [([277183.45492039167, 277198.46746226103], [5048847.323587381, 5048833.395284647]),
            ([277184.99787608377, 277199.4682983857], [5048847.782303939, 5048834.479523783])]
dist_pts2 = [([277388.21946463996, 277427.10081452655], [5049281.936758901, 5049254.797322375]),
            ([277385.8907249878, 277428.83142196125], [5049286.813559915, 5049257.285070563])]
dist_pts_list = [dist_pts1, dist_pts2]
offsets_1 = [(-20, -12), (7.9, 8.5)]
offsets_2 = [(-30, -2), (1, 8.2)]
offsets_list = [offsets_1, offsets_2]

for i, sel in enumerate([sel1, sel2]):
    x = x_mesh[sel, 0]
    y = x_mesh[sel, 1]

    X, Y = np.mgrid[int(x.min()):int(x.max()), int(y.min()):int(y.max())]

    h_up = U_up_mean[sel]
    h_ups_sig_up = U_up_mean[sel] + 2*U_ups_sig[sel]
    h_ups_sig_lo = U_up_mean[sel] - 2*U_ups_sig[sel]
    h_up_sig_up = U_up_mean[sel] + 2*U_up_sig[sel]
    h_up_sig_lo = U_up_mean[sel] - 2*U_up_sig[sel]

    H_up = griddata((x, y), h_up, (X, Y), method=method)
    H_ups_sig_up = griddata((x, y), h_ups_sig_up, (X, Y), method=method)
    H_ups_sig_lo = griddata((x, y), h_ups_sig_lo, (X, Y), method=method)
    H_up_sig_up = griddata((x, y), h_up_sig_up, (X, Y), method=method)
    H_up_sig_lo = griddata((x, y), h_up_sig_lo, (X, Y), method=method)

    ax = fig.add_subplot(gs[0, i])
    ax.contour(X, Y, H_up_sig_lo,
            levels=[h_0], alpha=0.3, colors=["C0"])
    cs = ax.contour(X, Y, H_up_sig_up,
            levels=[h_0], alpha=0.3, colors=["C0"])
    cs.collections[0].set_label(r"$\pm 2\sigma_\textrm{up}$")
    ax.contour(X, Y, H_ups_sig_lo,
            levels=[h_0], alpha=0.3, colors=["g"])
    cs = ax.contour(X, Y, H_ups_sig_up,
            levels=[h_0], alpha=0.3, colors=["g"])
    cs.collections[0].set_label(r"$\pm 2\sigma_\textrm{ups}$")
    cs = ax.contour(X, Y, H_up,
            levels=[h_0], colors=["b"])
    cs.collections[0].set_label(r"$\hat{u}_D$")
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r"$x\ [\textrm{m}]$")
    ax.set_ylabel(r"$y\ [\textrm{m}]$")
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))

    dist_pts_i = dist_pts_list[i]
    offsets = offsets_list[i]
    for i, dist_pts in enumerate(dist_pts_i):
        if i == 0:
            ax.plot(dist_pts[0], dist_pts[1], "k:", label="$d_{2\sigma}$")
        else:
            ax.plot(dist_pts[0], dist_pts[1], "k:")
        dist_i = np.sqrt((dist_pts[0][0] - dist_pts[0][1])**2 +
                           (dist_pts[1][0] - dist_pts[1][1])**2)
        print(offsets[i])
        ax.text(dist_pts[0][1] + offsets[i][0], dist_pts[1][1] + offsets[i][1],
                f"${dist_i:.2f}" + r"\ \textrm{m}$")
    ax.legend()

plt.tight_layout()
# plt.show()
savefig(os.path.join("results", f"podbnn-sw-zooms-up"), tight_box=True)
