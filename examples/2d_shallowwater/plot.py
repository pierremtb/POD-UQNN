"""Handles the plots for 2D inviscid Shallow Water Equations."""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re, re_mean_std
from podnn.plotting import figsize, saveresultdir


def plot_plot(fig, pos, x, y, z, z_min, z_max, title):
    """Does a colorplot from unstructured, 1d (x, y, z) data."""
    ax = fig.add_subplot(pos)
    h = plt.tripcolor(x, y, z)
    h.set_clim(z_min, z_max)
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


def plot_results(x_mesh, U_test, U_pred_mean, U_pred_std, sigma_pod,
                 resdir=None, train_res=None, HP=None,
                 export_vtk=False, export_txt=False):
    """Handles the plots and exports of 3d_shallowwater data."""

    x = x_mesh[:, 1]
    y = x_mesh[:, 2]

    U_test_mean = np.mean(U_test, axis=-1)
    U_test_std = np.nanstd(U_test, axis=-1)

    U_pred_mean = U_pred_mean[0]
    U_pred_std = U_pred_std[0]
    U_pred_mean_sig = U_pred_mean[1]
    U_pred_std_sig = U_pred_std[1]

    # Compute relative error
    error_test_mean, error_test_std = re_mean_std(U_test, U_pred)
    sigma_T = U_pred_mean_sig.mean()
    print(f"Test relative error: mean {error_test_mean:4f}, std {error_test_std:4f}")
    print(f"Mean Sigma on hifi predictions: {sigma_Thf:4f}")
    print(f"Mean Sigma contrib from POD: {sigma_pod:4f}")
    errors = {
        "REM_T": error_test_mean.item(),
        "RES_T": error_test_std.item(),
        "sigma": sigma_T.item(),
        "sigma_pod": sigma_pod.item(),
    }

    if export_txt:
        print("Saving to .txt")
        x_u_mean_std = np.concatenate((x_mesh, U_test_mean.T, U_test_std), axis=1)
        # x_u_std = np.concatenate((x_mesh, U_test_std.T), axis=1)
        non_idx_len = x_u_mean_std.shape[1] - 1
        np.savetxt(os.path.join(resdir, "x_u_mean_std.txt"), x_u_mean_std,
                   fmt=' '.join(["%i"] + ["%1.6f"]*non_idx_len),
                   delimiter="\t")
        # np.savetxt(os.path.join(resdir, "x_u_std.txt"), x_u_std,
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
        unstructuredGridToVTK(os.path.join(resdir, "x_u_mean_std"),
                              x, y, z,
                              connectivity, offsets, cell_types,
                              cellData=None,
                              pointData={
                                  "h_mean" : U_test_mean[0],
                                  "hu_mean" : U_test_mean[1],
                                  "hv_mean" : U_test_mean[2],
                                  "h_std" : U_test_std[0],
                                  "hu_std" : U_test_std[1],
                                  "hv_std" : U_test_std[2],
                                  })
        return

    print("Plotting")
    # Keeping only the first nodes
    i_min = 0
    i_max = 10000
    x = x[i_min:i_max]
    y = y[i_min:i_max]

    # Computing means
    U_test_mean = np.mean(U_test[:, i_min:i_max, :], axis=-1)
    U_pred_mean = np.mean(U_pred[:, i_min:i_max, :], axis=-1)
    U_test_std = np.nanstd(U_test[:, i_min:i_max, :], axis=-1)
    U_pred_std = np.nanstd(U_pred[:, i_min:i_max, :], axis=-1)
    n_plot_x = 4
    n_plot_y = 4
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.5))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    quantities = np.array([r"h", r"\eta", r"(hu)", r"(hv)"])
    idx_u = [i - 4 for i in HP["mesh_idx"][2]]
    for i, qty in enumerate(quantities[idx_u]):
        z_min, z_max = get_min_max(U_pred_mean[i], U_test_mean[i])
        plot_plot(fig, gs[0, i], x, y, U_pred_mean[i],
                  z_min, z_max, f"Mean ${qty}(x,y)$ [pred]")
        plot_plot(fig, gs[1, i], x, y, U_test_mean[i],
                  z_min, z_max, f"Mean ${qty}(x,y)$ [val]")
        z_min, z_max = get_min_max(U_pred_std[i], U_test_std[i])
        plot_plot(fig, gs[2, i], x, y, U_pred_std[i],
                  z_min, z_max, f"Std ${qty}(x,y)$ [pred]")
        plot_plot(fig, gs[3, i], x, y, U_test_std[i],
                  z_min, z_max, f"Std ${qty}(x,y)$ [val]")

    plt.tight_layout()
    saveresultdir(resdir, HP, errors, train_res)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise FileNotFoundError("Provide a resdir")

    resdir = sys.argv[1]

    with open(os.path.join(resdir, "HP.txt")) as HPFile:
        hp = yaml.load(HPFile)

    model = PodnnModel.load(resdir)

    x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))
    _, _, _, X_v_test, U_test = model.load_train_data()

    # Predict and restruct
    U_pred = model.predict(X_v_test)
    U_pred = model.restruct(U_pred)
    U_test = model.restruct(U_test)
    sigma_pod = model.sig_pod.mean()

    # Plot and save the results
    plot_results(x_mesh, U_test, U_pred, sigma_pod,
                 resdir=resdir, HP=hp, export_txt=True, export_vtk=True)
