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
from podnn.podnnmodel import PodnnModel
from podnn.plotting import figsize, saveresultdir
from podnn.metrics import re_mean_std, re
from podnn.mesh import read_space_sol_input_mesh


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


def plot_spec_time(fig, pos, x, U_pred, U_test,
                   title, show_legend=False):
    """1D plot at a specific time, u=f(x, t=t_i)."""
    ax = fig.add_subplot(pos)
    ax.plot(x, U_pred, "b-", label=r"$\hat{u_{T, hf}}$")
    ax.plot(x, U_test, "r--", label="$u_{T, hf}$")
    ax.set_title(title)
    ax.set_xlabel("$x'$")
    ax.set_title(title)
    if show_legend:
        ax.legend()


def get_min_max(z1, z2):
    """Returns the min and max across the two np array."""

    z_min = min([np.min(z1), np.min(z2)])
    z_max = max([np.max(z1), np.max(z2)])
    return z_min, z_max


def plot_results(csv_file, HP):
    print("Reading paraview results") results = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    x_line = results[:, 15]

    print("Plotting")
    n_plot_x = 4
    n_plot_y = 4
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.5))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    quantities = np.array([r"h", r"\eta", r"(hu)", r"(hv)"])
    idx_u = [i - 4 for i in HP["mesh_idx"][2]]
    for i, qty in enumerate(quantities[idx_u]):
        plot_spec_time(fig, gs[0, i], x_line, U_pred_hifi_mean[i], U_test_hifi_mean[i],
                       f"Means of  ${qty}(x,y)$")
        plot_spec_time(fig, gs[1, i], x_line, U_pred_hifi_std[i], U_test_hifi_std[i],
                       f"Standard devs ${qty}(x,y)$ [pred]")

    plt.tight_layout()
    saveresultdir(HP, train_res=None)


if __name__ == "__main__":
    from hyperparams import HP as hp

    # Plot and save the results
    plot_results(os.path.join("cache", "x_u_test_pred_mean_std.csv"), hp)
