"""Module for plotting results of 2D Shekel Equation."""

import os
import matplotlib.pyplot as plt
import numpy as np

from podnn.plotting import figsize, saveresultdir
from podnn.metrics import error_podnn
from podnn.testgenerator import X_FILE, U_MEAN_FILE, U_STD_FILE


def plot_contour(fig, pos, X, Y, U, levels, title):
    ax = fig.add_subplot(pos)
    ct = ax.contourf(X, Y, U, levels=levels, origin="lower")
    plt.colorbar(ct)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def get_test_data():
    dirname = os.path.join("data")
    X = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, U_test_mean, U_test_std


def plot_results(U, U_pred,
                 HP=None, no_plot=False):
    X, U_test_mean, U_test_std = get_test_data()
    X, Y = X[0], X[1]

    # Keeping the first coordinate only (n_v == 1)
    U_test_mean = U_test_mean[0, :, :]
    U_test_std = U_test_std[0, :, :]

    def restruct(U):
        return U.reshape(HP["n_x"], HP["n_y"], U.shape[-1])
    U_pred = restruct(U_pred)
    U = restruct(U)

    U_pred_mean = np.mean(U_pred, axis=-1)
    U_pred_std = np.std(U_pred, axis=-1)
    U_val_mean = np.mean(U, axis=-1)
    U_val_std = np.std(U, axis=-1)
    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    print(f"Error on the mean test HiFi LHS solution: {error_test_mean:.4f}%")
    print(f"Error on the stdd test HiFi LHS solution: {error_test_std:.4f}%")
    print("--")

    if no_plot:
        return error_test_mean, error_test_std

    mean_levels = list(range(2, 15))
    std_levels = np.arange(5, 20) * 0.1

    n_plot_x = 4
    n_plot_y = 6
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    plot_contour(fig, gs[0:2, 0:2],
                 X, Y, U_test_mean,
                 mean_levels, "Mean of $u_T$ (test)")
    plot_contour(fig, gs[0:2, 2:4],
                 X, Y, U_val_mean,
                 mean_levels, "Mean of $u_V$ (val)")
    plot_contour(fig, gs[0:2, 4:6],
                 X, Y, U_pred_mean,
                 mean_levels, "Mean of $\hat{u_V}$ (pred)")
    plot_contour(fig, gs[2:4, 0:2],
                 X, Y, U_test_std,
                 std_levels, "Standard deviation of $u_T$ (test)")
    plot_contour(fig, gs[2:4, 2:4],
                 X, Y, U_val_std,
                 std_levels, "Standard deviation of $u_V$ (val)")
    plot_contour(fig, gs[2:4, 4:6],
                    X, Y, U_pred_std,
                    std_levels, "Standard deviation of $\hat{u_V}$ (pred)")

    plt.tight_layout()

    saveresultdir(HP)

    return error_test_mean, error_test_std
