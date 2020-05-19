"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import pickle
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plti
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

sys.path.append(os.path.join("..", ".."))
from poduqnn.metrics import re_s
from poduqnn.plotting import figsize, savefig
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.mesh import read_multi_space_sol_input_mesh_txt

from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()
X_v_train_0, v_train_0, U_train_0, X_v_val_0, v_val_0, U_val_0 = model.load_init_data()

#%% Predict and restruct
# U_pred, U_pred_sig = model.predict(X_v_val)

# #%% Validation metrics
# U_pred, _ = model.predict(X_v_val)
# err_val = re_s(U_val, U_pred, div_max=True)
# print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
print(train_tst_idx)
# datadir = os.path.join("..", "..", "..", "scratch", "multi2swt") 
datadir = "data"
mu_path = os.path.join(datadir, "INPUT")
x_u_mesh_path = datadir
sel = np.loadtxt(os.path.join(datadir, "sel.csv"), skiprows=1, delimiter=",")[:, 0].astype("int64")
x_mesh, connectivity, X_v_tst, U_tst, points_idx = \
        read_multi_space_sol_input_mesh_txt(hp["n_s_tst"], hp["n_t"], hp["d_t"], train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"], sel)
bathymetry = meshio.read(os.path.join(datadir, f"multi_{train_tst_idx[1][0] + 1}", "0_FV-Paraview_0.vtk")).point_data["b"][points_idx]
print(bathymetry)
# X_v_tst = np.loadtxt(os.path.join("cache", "X_v_tst.txt"))
# U_tst = np.load(os.path.join("cache", "U_tst.npy"))
# print(U_tst.shape)
# U_tst_des = np.loadtxt(os.path.join("cache", "U_tst.txt"))
# connectivity = np.loadtxt(os.path.join("cache", "connectivity.txt"))
# x_mesh = np.loadtxt(os.path.join("cache", "x_mesh.txt"))
# U_tst = model.restruct(U_tst_des)
print("Elements count: ", connectivity.shape[0])
print("Nodes count: ", x_mesh.shape[0])
U_pred, U_pred_sig = model.predict(X_v_tst)
U_tst_des = model.destruct(U_tst)
err_val = re_s(U_tst_des, U_pred, div_max=True)
print(f"RE_tst: {err_val:4f}")

#%% VTU export
U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)
U_pred_up = U_pred + 2*U_pred_sig
U_pred_lo = U_pred - 2*U_pred_sig

U_pred_0 = model.project_to_U(model.project_to_v(U_tst_des))
U_pred_0 = model.restruct(U_pred_0)

print("Saving to .vtu")
for s in [0]:
    print(f"Sample is {X_v_tst[s*hp['n_t']][1]}")
    meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_{s}.{0}.vtu"),
                              x_mesh,
                              [("triangle", connectivity)],
                              point_data={
                                      "eta": U_tst[0, :, 0, s],
                                      "eta_pred": U_pred_0[0, :, 0, s],
                                      "eta_pred_up": U_pred_0[0, :, 0, s],
                                      "eta_pred_lo": U_pred_0[0, :, 0, s],
                              })
    for i in range(1, hp["n_t"] - 1):
        meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_{s}.{i}.vtu"),
                                x_mesh,
                                [("triangle", connectivity)],
                                point_data={
                                        "eta": U_tst[0, :, i, s],
                                        "eta_pred": U_pred[0, :, i, s],
                                        "eta_pred_up": U_pred_up[0, :, i, s],
                                        "eta_pred_lo": U_pred_lo[0, :, i, s],
                                })


#%% Cross section plotting
x = x_mesh[:, 0]
y = x_mesh[:, 1]
dxy = 1.
X, Y = np.mgrid[int(x.min()):int(x.max()):dxy, int(y.min()):int(y.max()):dxy]
method = "linear"
line = (
        [274805.820007385, 5043752.94918024],
        [274962.057873288, 5043861.33919971],
        )

# Load bathymetry
# b = np.loadtxt(os.path.join("cache", "b.csv"), delimiter=',', skiprows=1)[:, 5]

# Create coordinates from bathymethry line
num = 1000
line_x = np.linspace(line[0][0], line[1][0], num)
line_y = np.linspace(line[0][1], line[1][1], num)
line_X, line_Y = np.meshgrid(line_x, line_y)

def project(U):
    return np.diagonal(griddata((x, y), U, (line_X, line_Y), method=method))

x_prime_max = np.sqrt((line_x.max() - line_x.min())**2 + (line_y.max() - line_y.min())**2)
x_prime = np.linspace(0., x_prime_max, num)

b_ = project(bathymetry)


# Time samples
idx = [0, 5, 20, 100]
s = 0

# Custom loading
# for filename in os.listdir(os.path.join("cache", "azz")):
#     U_azzedine = np.loadtxt(os.path.join("cache", "azz", filename))
#     U_pred = U_azzedine[:, 0:1]
#     U_pred_lo = U_azzedine[:, 2:3]
#     U_pred_up = U_azzedine[:, 3:4]
#     s = 0
#     idx = [int(filename[-17]) * 10]
#     print(idx)
for _ in [1]:

    n_plot_x = 2*len(idx)
    n_plot_y = 5
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.0))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)
    ylim = (25.5, 31.5)
    for i, t_i in enumerate(idx):
        # Projections
        U_tst_ = project(U_tst[0, :, t_i, s])
        if i == 0:
            U_pred_ = project(U_pred_0[0, :, t_i, s])
            U_pred_lo_ = project(U_pred_0[0, :, t_i, s])
            U_pred_up_ = project(U_pred_0[0, :, t_i, s])
        else:
            U_pred_ = project(U_pred[0, :, t_i, s])
            U_pred_lo_ = project(U_pred_lo[0, :, t_i, s])
            U_pred_up_ = project(U_pred_up[0, :, t_i, s])
        # U_pred_ = project(U_pred[:, 0])
        # U_pred_lo_ = project(U_pred[:, 0])
        # U_pred_up_ = project(U_pred[:, 0])

        # Plot
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
        ax.fill_between(x_prime, np.zeros_like(b_), b_,
                        edgecolor="k", alpha=0.3, facecolor="w", hatch="/",
                        label="$b$")
        ax.plot(x_prime, U_pred_, "b-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
        ax.plot(x_prime, U_tst_, "r--", label=r"$u_D(s_{" + lbl + r"})$")
        ax.fill_between(x_prime, U_pred_lo_, U_pred_up_,
                        alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
        ax.set_xlabel(f"$x'$")
        ax.set_ylabel("$\eta$")
        ax.set_ylim(ylim)
        ax.set_title(f"$\eta_0={X_v_tst[s*hp['n_t']][1]:.3f}\ m$, $t={t_i * hp['d_t']}\ s$")
        if i == 0:
            ax.legend()
    plt.tight_layout()
    # plt.show()
    savefig("results/podensnn-swt-samples", True)
    # savefig(f"results/{filename}", True)
