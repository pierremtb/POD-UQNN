"""Handles the plots for 2D inviscid Shallow Water Equations."""

import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

sys.path.append(os.path.join("..", ".."))
from lib.podnnmodel import PodnnModel
from lib.plotting import figsize, saveresultdir
from lib.metrics import re_mean_std, re
from lib.mesh import read_space_sol_input_mesh


def plot_plot(fig, pos, x, y, z, z_min, z_max, title):
    """Does a colorplot from unstructured, 1d (x, y, z) data."""
    ax = fig.add_subplot(pos)
    h = plt.tripcolor(x, y, z)
    # h.set_clim(z_min, z_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def plot_spec_time(fig, pos, x, t_i, U_pred, U_test,
                   title, show_legend=False):
    """1D plot at a specific time, u=f(x, t=t_i)."""
    ax = fig.add_subplot(pos)
    ax.plot(x, U_pred[:, t_i], "b-", label=r"$\hat{u_V}$")
    ax.plot(x, U_test[:, t_i], "k,", label="$u_T$")
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_title(title)
    if show_legend:
        ax.legend()


def get_min_max(z1, z2):
    """Returns the min and max across the two np array."""

    z_min = min([np.min(z1), np.min(z2)])
    z_max = max([np.max(z1), np.max(z2)])
    return z_min, z_max


def export(x_mesh, U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                 U_test_hifi_mean, U_test_hifi_std,
                 train_res=None, HP=None,
                 export_vtk=False, export_txt=False):
    """Handles the plots and exports of 3d_shallowwater data."""

    x = x_mesh[:, 1]
    y = x_mesh[:, 2]

    if export_txt:
        print("Saving to .txt")
        x_u_mean_std = np.concatenate((x_mesh, U_pred_hifi_mean.T, U_pred_hifi_std), axis=1)
        # x_u_std = np.concatenate((x_mesh, U_test_std.T), axis=1)
        non_idx_len = x_u_mean_std.shape[1] - 1
        np.savetxt(os.path.join("cache", "x_u_mean_std.txt"), x_u_mean_std,
                   fmt=' '.join(["%i"] + ["%1.6f"]*non_idx_len),
                   delimiter="\t")
        # np.savetxt(os.path.join("cache", "x_u_std.txt"), x_u_std,
        #            fmt=' '.join(["%i"] + ["%1.6f"]*non_idx_len),
        #            delimiter="\t")
        if not export_vtk:
            return

    if export_vtk:
        print("Saving to .vtu")

        # Retrieving the mesh
        connectivity_raw = np.loadtxt(os.path.join("data", "connectivity.txt"))
        n_element = connectivity_raw.shape[0]

        # 1D list of connections
        connectivity = connectivity_raw[:, 1:4].astype("int64").flatten() - 1

        # 1d list of "offsets", ie. the end of each element
        # Since we use triangles, size = 3
        offsets = np.arange(1, n_element + 1) * 3
        cell_types = np.ones(n_element, dtype="int64") * VtkTriangle.tid
    
        # h_mean_el = np.zeros((n_element,))
        # for i in range(n_element):
        #     end = offsets[i]
        #     nodes = connectivity[end-3:end]
        #     values = U_pred_mean[0][nodes]
        #     h_mean_el[i] = values.mean()

        # Space points
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        z = np.ascontiguousarray(np.zeros_like(x))

        # Exporting
        unstructuredGridToVTK(os.path.join("cache", "x_u_test_pred_mean_std"),
                              x, y, z,
                              connectivity, offsets, cell_types,
                              cellData=None,
                              pointData={
                                  "h_mean" : U_test_hifi_mean[0],
                                  "hu_mean" : U_test_hifi_mean[1],
                                  "hv_mean" : U_test_hifi_mean[2],
                                  "h_std" : U_test_hifi_std[0],
                                  "hu_std" : U_test_hifi_std[1],
                                  "hv_std" : U_test_hifi_std[2],
                                  "h_mean_pred" : U_pred_hifi_mean[0],
                                  "hu_mean_pred" : U_pred_hifi_mean[1],
                                  "hv_mean_pred" : U_pred_hifi_mean[2],
                                  "h_std_pred" : U_pred_hifi_std[0],
                                  "hu_std_pred" : U_pred_hifi_std[1],
                                  "hv_std_pred" : U_pred_hifi_std[2],
                                  })
        return


if __name__ == "__main__":
    from hyperparams import HP as hp

    model = PodnnModel.load("cache")

    x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
    _, _, X_v_test, _, U_test = model.load_train_data()

    # Predict and restruct
    U_pred = model.predict(X_v_test)
    U_pred = model.restruct(U_pred)

    mu_path = os.path.join("data", f"INPUT_{hp['n_s_tst']}_Scenarios.txt")
    x_u_mesh_path = os.path.join("data", f"SOL_FV_{hp['n_s_tst']}_Scenarios.txt")
    _, u_mesh_test_hifi, X_v_test_hifi = \
        read_space_sol_input_mesh(hp["n_s"], hp["mesh_idx"], x_u_mesh_path, mu_path)
    U_test_hifi = model.u_mesh_to_U(u_mesh_test_hifi, hp["n_s_tst"])
    U_test_hifi_mean, U_test_hifi_std = U_test_hifi.mean(-1), np.nanstd(U_test_hifi, -1)

    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)
    error_test_hifi_mean = re(U_pred_hifi_mean, U_test_hifi_mean)
    error_test_hifi_std = re(U_pred_hifi_std, U_test_hifi_std)
    print(f"Hifi Test relative error: mean {error_test_hifi_mean:4f}, std {error_test_hifi_std:4f}")

    # Restruct for plotting
    U_test_hifi_mean = model.restruct(U_test_hifi_mean, no_s=True)
    U_test_hifi_std = model.restruct(U_test_hifi_std, no_s=True)
    U_pred_hifi_mean = model.restruct(U_pred_hifi_mean, no_s=True)
    U_pred_hifi_std = model.restruct(U_pred_hifi_std, no_s=True)

    # Plot and save the results
    export(x_mesh, U_pred, U_pred_hifi_mean, U_pred_hifi_std,
           U_test_hifi_mean, U_test_hifi_std,
           train_res=None, HP=hp, export_vtk=False, export_txt=False)
