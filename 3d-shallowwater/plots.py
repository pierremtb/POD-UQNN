import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import sys
import os
from tqdm import tqdm
import json
import time

eqnPath = "3d-shallowwater"
sys.path.append(eqnPath)
from hyperparams import hp

sys.path.append("utils")
from metrics import error_podnn
from plotting import figsize, saveresultdir, savefig
from handling import pack_layers


def plot_contour(fig, pos, X, T, U, levels, title):
    ax = fig.add_subplot(pos)
    ct = ax.contourf(X, T, U, origin="lower")
    # ct = ax.contourf(X, T, U, levels=levels, origin="lower")
    plt.colorbar(ct)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")


def plot_map(fig, pos, x, t, X, T, U, title):
    XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    U_test_grid = griddata(XT, U.flatten(), (X, T), method='cubic')
    ax = fig.add_subplot(pos)
    h = ax.imshow(U_test_grid, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
            origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title(title)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")


def plot_plot(fig, pos, x, y, z, z_min, z_max, title):
    ax = fig.add_subplot(pos)
    # h = plt.scatter(x, y, c=z, linewidths=0)
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
    z_min = min([np.min(z1), np.min(z2)])
    z_max = max([np.max(z1), np.max(z2)])
    return z_min, z_max


def plot_results(x_mesh, U_val, U_pred,
                 hp=None, save_path=None):
    lim = 1000
    x = x_mesh[:lim, 1]
    y = x_mesh[:lim, 2]
    # yy, xx = np.meshgrid(x, y)

    print("Computing means")
    U_val_mean = np.mean(U_val[:lim, :, :], axis=-1)
    U_pred_mean = np.mean(U_pred[:lim, :, :], axis=-1)
    # U_pred_mean = np.mean(U_pred[:lim, :], axis=-1)
    # U_val_mean = np.mean(U_val[:lim, :], axis=-1)
    # U_val_mean = U_val[:lim, 0]

    print("Plotting")
    z = U_val_mean[:, 0]

    # plt.scatter(y, U_val_mean[:, 1], c="g", marker=".")
    # plt.scatter(y, U_val_mean[:, 0], c="b", marker=",")
    # plt.scatter(y, U_pred_mean[:, 0], c="r", marker=",")
    # plt.show()

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
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()
