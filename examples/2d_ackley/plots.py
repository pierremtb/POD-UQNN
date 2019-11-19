"""Module for plotting results of 2D Shekel Equation."""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
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

def plot_slice(fig, pos, x, u_pred, u_val, u_test, u_hifi, title):
    ax = fig.add_subplot(pos)
    ax.plot(x, u_pred, "b-", label="$\hat{u_V}$")
    ax.plot(x, u_val, "r--", label="$u_V$")
    ax.plot(x, u_test, "k,", label="$u_T$")
    ax.plot(x, u_hifi, "b,", label="$\hat{u_T}$")
    ax.set_xlabel("$x$")
    ax.set_title(title)
    ax.legend()

def get_test_data():
    dirname = os.path.join("data")
    X = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, U_test_mean, U_test_std


def plot_results(U, U_pred, U_pred_hifi,
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

    U_pred_hifi_mean = np.mean(U_pred_hifi, axis=-1)
    U_pred_hifi_std = np.std(U_pred_hifi, axis=-1)
    hifi_error_test_mean = 100 * error_podnn(U_test_mean, U_pred_hifi_mean)
    hifi_error_test_std = 100 * error_podnn(U_test_std, U_pred_hifi_std)

    print(f"Error on the mean test HiFi LHS solution: {error_test_mean:.4f}%")
    print(f"Error on the stdd test HiFi LHS solution: {error_test_std:.4f}%")
    print("--")

    print(f"HiFi Error on the mean test HiFi LHS solution: {hifi_error_test_mean:.4f}%")
    print(f"HiFi Error on the stdd test HiFi LHS solution: {hifi_error_test_std:.4f}%")
    print("--")

    if no_plot:
        return error_test_mean, error_test_std

    mean_levels = list(range(2, 15))
    std_levels = np.arange(5, 20) * 0.1

    n_plot_x = 8
    n_plot_y = 8
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)
    x = X[199, :]
    plot_slice(fig, gs[0:4, 0:4], x,
               U_pred_mean[:, 199], U_val_mean[:, 199],
               U_test_mean[:, 199], U_pred_hifi_mean[:, 199], "Means $u(x, y=0)$") 
    plot_slice(fig, gs[4:, 0:4], x,
               U_pred_std[:, 199], U_val_std[:, 199],
               U_test_std[:, 199], U_pred_hifi_std[:, 199], "Std dev $u(x, y=0)$") 
    plot_slice(fig, gs[0:4, 4:8], x,
               U_pred_mean[199, :], U_val_mean[199, :],
               U_test_mean[199, :], U_pred_hifi_mean[199, :], "Means $u(x=0, y)$") 
    plot_slice(fig, gs[4:, 4:], x,
               U_pred_std[199, :], U_val_std[199, :],
               U_test_std[199, :], U_pred_hifi_std[199, :], "Std dev $u(x=0, y)$") 

    # plot_contour(fig, gs[0:2, 0:2],
    #              X, Y, U_test_mean,
    #              mean_levels, "Mean of $u_T$ (test)")
    # plot_contour(fig, gs[0:2, 2:4],
    #              X, Y, U_val_mean,
    #              mean_levels, "Mean of $u_V$ (val)")
    # plot_contour(fig, gs[0:2, 4:6],
    #              X, Y, U_pred_mean,
    #              mean_levels, "Mean of $\hat{u_V}$ (pred)")
    # plot_contour(fig, gs[2:4, 0:2],
    #              X, Y, U_test_std,
    #              std_levels, "Standard deviation of $u_T$ (test)")
    # plot_contour(fig, gs[2:4, 2:4],
    #              X, Y, U_val_std,
    #              std_levels, "Standard deviation of $u_V$ (val)")
    # plot_contour(fig, gs[2:4, 4:6],
    #                 X, Y, U_pred_std,
    #                 std_levels, "Standard deviation of $\hat{u_V}$ (pred)")

    plt.tight_layout()

    saveresultdir(HP)

    return error_test_mean, error_test_std

if __name__ == "__main__":
    from hyperparams import HP as hp

    model = PodnnModel.load("cache")

    x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
    _, _, X_v_val, _, U_val = model.load_train_data()

    # Predict and restruct
    U_pred = model.predict(X_v_val)

    # Plot and save the results
    plot_results(U_val, U_pred, hp)