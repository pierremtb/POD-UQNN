"""Handles the plots for 3D time-steady Shallow Water Equations."""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

EQN_PATH = "3d-shallowwater"
sys.path.append(EQN_PATH)

sys.path.append("utils")
from metrics import error_podnn
from plotting import figsize, saveresultdir, savefig
from handling import pack_layers


def plot_plot(fig, pos, x, y, z, z_min, z_max, title):
    """Does a colorplot from unstructured, 1d (x, y, z) data."""

    ax = fig.add_subplot(pos)
    h = plt.tripcolor(x, y ,z)
    h.set_clim(z_min, z_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def plot_spec_time(fig, pos, x, t_i, U_pred, U_val, U_test,
                   title, show_legend=False):
    ax = fig.add_subplot(pos)
    ax.plot(x, U_pred[:, t_i], "b-", label="$\hat{u_V}$")
    ax.plot(x, U_val[:, t_i], "r--", label="$u_V$")
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


def plot_results(x_mesh, U_val, U_pred,
                 HP=None, save_path=None):
    """Handles the plots of 3d-shallowwater."""
   
    # Keeping only the first nodes
    lim = 1000
    x = x_mesh[:lim, 1]
    y = x_mesh[:lim, 2]

    # Computing means
    U_val_mean = np.mean(U_val[:lim, :, :], axis=-1)
    U_pred_mean = np.mean(U_pred[:lim, :, :], axis=-1)

    print("Plotting")
    n_plot_x = 2
    n_plot_y = 4
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.5))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    quantities = ["h", "\eta", "(hu)", "(hv)"]
    for i, qty in enumerate(quantities):
        z_min, z_max = get_min_max(U_pred_mean[:, i], U_val_mean[:, i])
        plot_plot(fig, gs[0, i], x, y, U_pred_mean[:, i],
                  z_min, z_max, f"Mean ${qty}(x,y)$ [pred]")
        plot_plot(fig, gs[1, i], x, y, U_val_mean[:, i],
                  z_min, z_max, f"Mean ${qty}(x,y)$ [val]")

    plt.tight_layout()
    if save_path is not None:
        saveresultdir(save_path, save_HP=HP)
    else:
        plt.show()
