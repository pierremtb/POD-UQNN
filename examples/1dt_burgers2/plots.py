"""Module for plotting results of 3D time-dependante Burgers Equation."""

import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from podnn.plotting import figsize, saveresultdir
from podnn.metrics import error_podnn
from podnn.testgenerator import X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE


def get_test_data():
    dirname = os.path.join("data")
    X = np.load(os.path.join(dirname, X_FILE))
    T = np.load(os.path.join(dirname, T_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, T, U_test_mean, U_test_std


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


def plot_results(U_val, U_pred,
                 HP=None, save_path=None):
    X, t, U_test_mean, U_test_std = get_test_data()
    print(t.shape)
    x = X[0]
    tt, xx = np.meshgrid(x, t)

    U_pred_mean = np.mean(U_pred, axis=2)
    U_val_mean = np.mean(U_val, axis=2)
    # Using nanstd() to prevent NotANumbers from appearing
    # (they prevent norm to be computed after)
    U_pred_std = np.nanstd(U_pred, axis=2)
    U_val_std = np.nanstd(U_val, axis=2)
    U_test_std = np.nan_to_num(U_test_std)

    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    if save_path is None:
        print("--")
        print(f"Error on the mean test HiFi LHS solution: {error_test_mean:.4f}%")
        print(f"Error on the stdd test HiFi LHS solution: {error_test_std:.4f}%")
        print("--")

    n_plot_x = 5
    n_plot_y = 3
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.5))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    plot_map(fig, gs[0, :n_plot_y], x, t, xx, tt, U_pred_mean, "Mean $u(x,t)$ [pred]")
    plot_map(fig, gs[1, :n_plot_y], x, t, xx, tt, U_test_mean, "Mean $u(x,t)$ [test]")
    plot_spec_time(fig, gs[2, 0], x, 25, 
            U_pred_mean, U_val_mean, U_test_mean,
            "Means $u(x, t=0.25)$", show_legend=True)
    plot_spec_time(fig, gs[2, 1], x, 50,
            U_pred_mean, U_val_mean, U_test_mean, "Means $u(x, t=0.50)$")
    plot_spec_time(fig, gs[2, 2], x, 75,
            U_pred_mean, U_val_mean, U_test_mean, "Means $u(x, t=0.75)$")
    plot_spec_time(fig, gs[3, 0], x, 25,
            U_pred_std, U_val_std, U_test_std, "Std dev $u(x, t=0.25)$")
    plot_spec_time(fig, gs[3, 1], x, 50,
            U_pred_std, U_val_std, U_test_std, "Std dev $u(x, t=0.50)$")
    plot_spec_time(fig, gs[3, 2], x, 75,
            U_pred_std, U_val_std, U_test_std, "Std dev $u(x, t=0.75)$")

    plt.tight_layout()
    if save_path is not None:
        saveresultdir(HP)
    else:
        plt.show()
