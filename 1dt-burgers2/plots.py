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

eqnPath = "1dt-burgers2"
sys.path.append(eqnPath)
from hyperparams import hp

sys.path.append("utils")
from metrics import error_podnn
from plotting import figsize, saveresultdir, savefig
from handling import pack_layers


def get_test_data():
    from datagen import X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE
    dirname = os.path.join(eqnPath, "data")
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
                 hp=None, save_path=None):
    X, T, U_test_mean, U_test_std = get_test_data()
    t = T[0, :]
    x = X[:, 0]

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

    plot_map(fig, gs[0, :n_plot_y], x, t, X, T, U_pred_mean, "Mean $u(x,t)$ [pred]")
    plot_map(fig, gs[1, :n_plot_y], x, t, X, T, U_test_mean, "Mean $u(x,t)$ [test]")
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
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()


# if __name__ == "__main__":
#     X_v_train, v_train, X_v_val, v_val, \
#         lb, ub, V, U_val = prep_data(hp, use_cache=True)
        
#     hp["layers"] = pack_layers(X_v_train.shape[1], hp["h_layers"],
#                                X_v_train.shape[1])
#     regnn = RegNN.load_from(os.path.join(eqnPath, "cache", "model.h5"),
#                        hp, lb, ub)

#     U_val_struct, U_pred_struct = predict_and_assess(regnn, X_v_val, U_val,
#                                                      V, hp)

#     plot_results(U_val_struct, U_pred_struct, hp)
