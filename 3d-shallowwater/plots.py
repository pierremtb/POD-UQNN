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


def plot_results(x_mesh, U_val, U_pred,
                 hp=None, save_path=None):
    x = x_mesh[:, 1]
    y = x_mesh[:, 2]
    print(x.shape, y.shape)
    Xt, Yt = np.meshgrid(x, y)
    X, Y = Xt.T, Yt.T
    print(X.shape, Y.shape)
    x = X[:, 0]
    y = Y[0, :]
    print(U_val.shape)
    print(U_pred.shape)
    exit(0)
    U_pred_mean = np.mean(U_pred, axis=-1)
    U_val_mean = np.mean(U_val, axis=-1)
    # Using nanstd() to prevent NotANumbers from appearing
    # (they prevent norm to be computed after)
    U_pred_std = np.nanstd(U_pred, axis=-1)
    U_val_std = np.nanstd(U_val, axis=-1)

    n_plot_x = 2
    n_plot_y = 1
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.5))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    plot_map(fig, gs[0, :n_plot_y], x, t, X, T, U_pred_mean, "Mean $u(x,t)$ [pred]")
    plot_map(fig, gs[1, :n_plot_y], x, t, X, T, U_test_mean, "Mean $u(x,t)$ [test]")

    plt.tight_layout()
    plt.show()
    exit(0)
    if save_path is not None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()
