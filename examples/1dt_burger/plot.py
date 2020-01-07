"""Module for plotting results of the 1d time-dep Burgers Equation."""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.plotting import figsize, saveresultdir
from podnn.metrics import re
from podnn.testgenerator import X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE


def get_test_data():
    dirname = os.path.join("data")
    X = np.load(os.path.join(dirname, X_FILE))
    T = np.load(os.path.join(dirname, T_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, T, U_test_mean, U_test_std


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


def plot_spec_time(fig, pos, x, t_i,
                   U_pred, U_pred_hifi, U_pred_hifi_sig, U_test_hifi,
                   title, show_legend=False):
    ax = fig.add_subplot(pos)
    ax.plot(x, U_pred[0, :, t_i], "k,", label="$\hat{u_T}$")
    ax.plot(x, U_pred_hifi[0, :, t_i], "b-", label="$\hat{u_{T,hf}}$")
    ax.plot(x, U_test_hifi[0, :, t_i], "r--", label="$u_{T,hf}$")
    lower = U_pred_hifi[0, :, t_i] - 2.0*U_pred_hifi_sig[0, :, t_i]
    upper = U_pred_hifi[0, :, t_i] + 2.0*U_pred_hifi_sig[0, :, t_i]
    plt.fill_between(x, lower, upper, 
                     facecolor='orange', alpha=0.5, label=r"$\sigma_{T,hf}(x)$")
    ax.set_xlabel("$x$")
    ax.set_title(title)
    if show_legend:
        ax.legend()


def plot_results(U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                 train_res=None, HP=None, no_plot=False):
    X, t, U_test_hifi_mean, U_test_hifi_std = get_test_data()
    x = X[0]

    xxT, ttT = np.meshgrid(x, t)
    xx = xxT.T
    tt = ttT.T

    U_pred_mean = U_pred.mean(-1)
    # Using nanstd() to prevent NotANumbers from appearing
    U_pred_std = U_pred.std(-1)

    U_pred_hifi_mean_sig = U_pred_hifi_mean[1]
    U_pred_hifi_std_sig = U_pred_hifi_std[1]

    U_pred_hifi_mean = U_pred_hifi_mean[0]
    U_pred_hifi_std = U_pred_hifi_std[0]

    hifi_error_test_mean = re(U_test_hifi_mean, U_pred_hifi_mean)
    hifi_error_test_std = re(U_test_hifi_std, U_pred_hifi_std)
    print(f"HiFi test relative error: mean {hifi_error_test_mean:4f}, std {hifi_error_test_std:4f}")

    n_plot_x = 5
    n_plot_y = 3
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.5))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    plot_map(fig, gs[0, :n_plot_y], x, t, xx, tt, U_pred_hifi_mean, "Mean $u(x,t)$ [pred]")
    plot_map(fig, gs[1, :n_plot_y], x, t, xx, tt, U_test_hifi_mean, "Mean $u(x,t)$ [test]")
    plot_spec_time(fig, gs[2, 0], x, 25, 
                   U_pred_mean, U_pred_hifi_mean, U_pred_hifi_mean_sig, U_test_hifi_mean,
                   "Means $u(x, t=0.25)$", show_legend=True)
    plot_spec_time(fig, gs[2, 1], x, 50,
                   U_pred_mean, U_pred_hifi_mean, U_pred_hifi_mean_sig, U_test_hifi_mean,
                   "Means $u(x, t=0.50)$")
    plot_spec_time(fig, gs[2, 2], x, 75,
                   U_pred_mean, U_pred_hifi_mean, U_pred_hifi_mean_sig, U_test_hifi_mean,
                   "Means $u(x, t=0.75)$")
    plot_spec_time(fig, gs[3, 0], x, 25,
                   U_pred_std, U_pred_hifi_std, U_pred_hifi_std_sig, U_test_hifi_std,
                   "Std dev $u(x, t=0.25)$")
    plot_spec_time(fig, gs[3, 1], x, 50,
                   U_pred_std, U_pred_hifi_std, U_pred_hifi_std_sig, U_test_hifi_std,
                   "Std dev $u(x, t=0.50)$")
    plot_spec_time(fig, gs[3, 2], x, 75,
                   U_pred_std, U_pred_hifi_std, U_pred_hifi_std_sig, U_test_hifi_std,
                   "Std dev $u(x, t=0.75)$")

    plt.tight_layout()
    saveresultdir(HP, train_res)


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
    X_v_test_hifi = model.generate_hifi_inputs(n_s_hifi, hp["mu_min"], hp["mu_max"],
                                               hp["t_min"], hp["t_max"])
    print("Predicting the {n_s_hifi} corresponding solutions")
    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)

    # Plot and save the results
    plot_results(U_pred_struct, U_pred_hifi_mean, U_pred_hifi_std, HP=hp)
