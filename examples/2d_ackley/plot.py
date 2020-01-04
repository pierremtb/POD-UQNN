"""Module for plotting results of 2D Shekel Equation."""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.plotting import figsize, saveresultdir
from podnn.metrics import re
from podnn.testgenerator import X_FILE, U_MEAN_FILE, U_STD_FILE

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


def plot_contour(fig, pos, X, Y, U, levels, title):
    ax = fig.add_subplot(pos)
    ct = ax.contourf(X, Y, U, levels=levels, origin="lower")
    plt.colorbar(ct)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

def plot_slice(fig, pos, x, U_pred, U_pred_hifi, U_pred_hifi_sig, U_test_hifi, title, legend=False):
    ax = fig.add_subplot(pos)
    ax.plot(x, U_pred, "k,", label=r"$\hat{u}_T(x)$")
    ax.plot(x, U_pred_hifi, "b-", label=r"$\hat{u}_{T,hf}(x)$")
    ax.plot(x, U_test_hifi, "r--", label=r"$u_{T,hf}(x)$")
    lower = U_pred_hifi - 2.0*U_pred_hifi_sig
    upper = U_pred_hifi + 2.0*U_pred_hifi_sig
    plt.fill_between(x, lower, upper, 
                     facecolor='orange', alpha=0.5, label=r"2*std")
    ax.set_xlabel("$x$")
    ax.set_title(title)
    if legend:
        ax.legend()

def get_test_data():
    dirname = os.path.join("data")
    X = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, U_test_mean, U_test_std

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

def plot_results(U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                 train_res=None, HP=None, no_plot=False):
    X, U_test_hifi_mean, U_test_hifi_std = get_test_data()
    X, Y = X[0], X[1]

    u_shape = (HP["n_v"], HP["n_x"], HP["n_y"])

    U_pred_mean = np.mean(U_pred, axis=-1).reshape(u_shape)
    # Using nanstd() to prevent NotANumbers from appearing
    U_pred_std = np.nanstd(U_pred, axis=-1).reshape(u_shape)

    U_pred_hifi_mean_sig = U_pred_hifi_mean[1]
    U_pred_hifi_std_sig = U_pred_hifi_std[1]

    U_pred_hifi_mean = U_pred_hifi_mean[0]
    U_pred_hifi_std = U_pred_hifi_std[0]

    U_pred_hifi_mean = U_pred_hifi_mean.reshape(u_shape)
    U_pred_hifi_std = U_pred_hifi_std.reshape(u_shape)
    U_pred_hifi_mean_sig = U_pred_hifi_mean_sig.reshape(u_shape)
    U_pred_hifi_std_sig = U_pred_hifi_std_sig.reshape(u_shape)
    U_test_hifi_mean = U_test_hifi_mean.reshape(u_shape)
    U_test_hifi_std = U_test_hifi_std.reshape(u_shape)

    hifi_error_test_mean = re(U_test_hifi_mean, U_pred_hifi_mean)
    hifi_error_test_std = re(U_test_hifi_std, U_pred_hifi_std)
    print(f"HiFi test relative error: mean {hifi_error_test_mean:4f}, std {hifi_error_test_std:4f}")

    if no_plot:
        return hifi_error_test_mean, hifi_error_test_std


    n_plot_x = 3
    n_plot_y = 2
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)
    x = X[199, :]
    y = Y[:, 199]
    plot_map(fig, gs[0, 0],
             x, y, X, Y, U_pred_hifi_mean[0],
             r"Mean of $\hat{u}_{T,hf}$")
    plot_map(fig, gs[0, 1],
             x, y, X, Y, U_test_hifi_mean[0],
             r"Mean of $u_{T,hf}$")
    plot_slice(fig, gs[1, 0], x,
               U_pred_mean[0, :, 199], U_pred_hifi_mean[0, :, 199], U_pred_hifi_mean_sig[0, :, 199],
               U_test_hifi_mean[0, :, 199], "Means $u(x, y=0)$", legend=True) 
    plot_slice(fig, gs[1, 1], x,
               U_pred_std[0, :, 199], U_pred_hifi_std[0, :, 199], U_pred_hifi_std_sig[0, :, 199],
               U_test_hifi_std[0, :, 199], "Std dev $u(x, y=0)$") 
    plot_slice(fig, gs[2, 0], y,
               U_pred_mean[0, 199, :], U_pred_hifi_mean[0, 199, :], U_pred_hifi_mean_sig[0, 199, :],
               U_test_hifi_mean[0, 199, :], "Means $u(x=0, y)$") 
    plot_slice(fig, gs[2, 1], y,
               U_pred_std[0, 199, :], U_pred_hifi_std[0, 199, :], U_pred_hifi_std_sig[0, 199, :],
               U_test_hifi_std[0, 199, :], "Std dev $u(x=0, y)$") 

    # mean_levels = list(range(2, 15))

    # plot_contour(fig, gs[0, 0],
    #              X, Y, U_pred_hifi_mean[0],
    #              mean_levels, r"Mean of $\hat{u}_{T,hf}$")
    # plot_contour(fig, gs[0, 1],
    #              X, Y, U_test_hifi_mean[0],
    #              mean_levels, r"Mean of $u_{T,hf}$")

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
    print("Sampling {n_s_hifi} parameters")
    X_v_test_hifi = model.generate_hifi_inputs(n_s_hifi, hp["mu_min"], hp["mu_max"])
    print("Predicting the {n_s_hifi} corresponding solutions")
    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)

    # Plot and save the results
    plot_results(U_pred_struct, U_pred_hifi_mean, U_pred_hifi_std, HP=hp)
