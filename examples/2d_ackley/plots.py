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

def plot_slice(fig, pos, x, u_pred, u_pred_hifi, u_test_hifi, title):
    ax = fig.add_subplot(pos)
    ax.plot(x, u_pred, "b,", label=r"$\hat{u_T}(x)$")
    ax.plot(x, u_pred_hifi, "b-", label=r"$\hat{u_T^{hf}}(x)$")
    ax.plot(x, u_test_hifi, "r--", label=r"$u_T^{hf}(x)$")
    ax.set_xlabel("$x$")
    ax.set_title(title)
    ax.legend()

def get_test_data():
    dirname = os.path.join("data")
    X = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, U_test_mean, U_test_std


def plot_results(U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                 train_res=None, HP=None, no_plot=False):
    X, U_test_hifi_mean, U_test_hifi_std = get_test_data()
    X, Y = X[0], X[1]

    U_pred_mean = np.mean(U_pred, axis=-1)
    # Using nanstd() to prevent NotANumbers from appearing
    U_pred_std = np.nanstd(U_pred, axis=-1)

    hifi_error_test_mean = error_podnn(U_test_hifi_mean, U_pred_hifi_mean)
    hifi_error_test_std = error_podnn(U_test_hifi_std, U_pred_hifi_std)
    print(f"HiFi test relative error: mean {hifi_error_test_mean:4f}, std {hifi_error_test_std:4f}")

    if no_plot:
        return hifi_error_test_mean, hifi_error_test_std

    # mean_levels = list(range(2, 15))
    # std_levels = np.arange(5, 20) * 0.1

    n_plot_x = 8
    n_plot_y = 8
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)
    x = X[199, :]
    y = Y[199, :]
    plot_slice(fig, gs[0:4, 0:4], x,
               U_pred_mean[0, :, 199], U_pred_hifi_mean[0, :, 199],
               U_test_hifi_mean[0, :, 199], "Means $u(x, y=0)$") 
    plot_slice(fig, gs[4:, 0:4], x,
               U_pred_std[0, :, 199], U_pred_hifi_std[0, :, 199],
               U_test_hifi_std[0, :, 199], "Std dev $u(x, y=0)$") 
    plot_slice(fig, gs[0:4, 4:8], y,
               U_pred_mean[0, 199, :], U_pred_hifi_mean[0, 199, :],
               U_test_hifi_mean[0, 199, :], "Means $u(x=0, y)$") 
    plot_slice(fig, gs[4:, 4:], y,
               U_pred_std[0, 199, :], U_pred_hifi_std[0, 199, :],
               U_test_hifi_std[0, 199, :], "Std dev $u(x=0, y)$") 

    # plot_contour(fig, gs[0:2, 0:2],
    #              X, Y, U_test_mean,
    #              mean_levels, "Mean of $u_T$ (test)")
    # plot_contour(fig, gs[0:2, 2:4],
    #              X, Y, U_test_mean,
    #              mean_levels, "Mean of $u_V$ (val)")
    # plot_contour(fig, gs[0:2, 4:6],
    #              X, Y, U_pred_mean,
    #              mean_levels, "Mean of $\hat{u_V}$ (pred)")
    # plot_contour(fig, gs[2:4, 0:2],
    #              X, Y, U_test_std,
    #              std_levels, "Standard deviation of $u_T$ (test)")
    # plot_contour(fig, gs[2:4, 2:4],
    #              X, Y, U_test_std,
    #              std_levels, "Standard deviation of $u_V$ (val)")
    # plot_contour(fig, gs[2:4, 4:6],
    #                 X, Y, U_pred_std,
    #                 std_levels, "Standard deviation of $\hat{u_V}$ (pred)")

    plt.tight_layout()

    saveresultdir(HP, train_res)

    return hifi_error_test_mean, hifi_error_test_std

if __name__ == "__main__":
    from hyperparams import HP as hp

    model = PodnnModel.load("cache")

    x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
    _, _, X_v_test, _, U_test = model.load_train_data()

    # Predict and restruct
    U_pred = model.predict(X_v_test)
    U_pred_struct = model.restruct(U_pred)
    U_test_struct = model.restruct(U_test)

    # Sample the new model to generate a HiFi prediction
    n_s_hifi = hp["n_s_hifi"]
    print("Sampling {n_s_hifi} parameters...")
    X_v_test_hifi = model.generate_hifi_inputs(n_s_hifi, hp["mu_min"], hp["mu_max"],
                                               hp["t_min"], hp["t_max"])
    print("Predicting the {n_s_hifi} corresponding solutions...")
    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)

    # Plot and save the results
    plot_results(U_pred_struct, U_pred_hifi_mean, U_pred_hifi_std, HP=hp)
