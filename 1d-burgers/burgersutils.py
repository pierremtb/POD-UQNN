from metrics import error_podnn
from plotting import figsize, saveresultdir, savefig
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from pyDOE import lhs
from deap.benchmarks import shekel
import json

eqnPath = "1d-burgers"
sys.path.append("utils")
sys.path.append(os.path.join("datagen", eqnPath))
sys.path.append(os.path.join(eqnPath, "burgersutils"))
from burgers import burgers_viscous_time_exact1 as burgers_u
from names import X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE


def restruct(U, n_x, n_t, n_s):
    return np.reshape(U, (n_x, n_t, n_s))


def prep_data(n_x, x_min, x_max, n_t, t_min, t_max, n_s, mu_mean):
    # Total number of snapshots
    nn_s = n_t*n_s

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling...")
    pbar = tqdm(total=100)
    X = lhs(n_s, 1).T
    pbar.update(50)
    lb = mu_mean * (1 - np.sqrt(3)/10)
    ub = mu_mean * (1 + np.sqrt(3)/10)
    mu_lhs = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()

    # Number of inputs in time plus number of parameters
    n_d = 1 + 1

    # Creating the snapshots
    print(f"Generating {nn_s} corresponding snapshots")
    U = np.zeros((n_x, nn_s))
    X_v = np.zeros((nn_s, n_d))
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    tT = t.reshape((n_t, 1))
    for i in tqdm(range(n_s)):
        # Calling the analytical solution function
        U[:, i:i+n_t] = burgers_u(mu_lhs[i, :], n_x, x, n_t, t)
        X_v[i:i+n_t, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))
    return U, X_v, lb, ub


def plot_contour(fig, pos, X, T, U, levels, title):
    ax = fig.add_subplot(pos)
    ct = ax.contourf(X, T, U, levels=levels, origin="lower")
    plt.colorbar(ct)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")


def get_test_data():
    dirname = os.path.join(eqnPath, "data")
    X = np.load(os.path.join(dirname, X_FILE))
    T = np.load(os.path.join(dirname, T_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, T, U_test_mean, U_test_std


def plot_results(U, U_pred=None,
                 hp=None, save_path=None):
    X, T, U_test_mean, U_test_std = get_test_data()

    print(U.shape)
    print(U_pred.shape)
    print(np.mean(U, axis=2).shape)

    U_pred_mean = np.mean(U_pred, axis=2)
    U_pred_std = np.std(U_pred, axis=2)
    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    if save_path is not None:
        print("--")
        print(f"Error on the mean test HiFi LHS solution: {error_test_mean:.4f}%")
        print(f"Error on the stdd test HiFi LHS solution: {error_test_std:.4f}%")
        print("--")

    mean_levels = list(range(2, 15))
    std_levels = np.arange(5, 20) * 0.1

    n_plot_x = 4
    n_plot_y = 6
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=1.))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    plot_contour(fig, gs[0:2, 0:2],
                 X, T, U_test_mean,
                 mean_levels, "Mean of $u_T$ (test)")
    plot_contour(fig, gs[0:2, 2:4],
                 X, T, np.mean(U, axis=2),
                 mean_levels, "Mean of $u_V$ (val)")
    if U_pred is not None:
        plot_contour(fig, gs[0:2, 4:6],
                     X, T, np.mean(U_pred, axis=2),
                     mean_levels, "Mean of $\hat{u_V}$ (pred)")
    plot_contour(fig, gs[2:4, 0:2],
                 X, T, U_test_std,
                 std_levels, "Standard deviation of $u_T$ (test)")
    plot_contour(fig, gs[2:4, 2:4],
                 X, T, np.std(U, axis=2),
                 std_levels, "Standard deviation of $u_V$ (val)")
    if U_pred is not None:
        plot_contour(fig, gs[2:4, 4:6],
                     X, T, np.std(U_pred, axis=2),
                     std_levels, "Standard deviation of $\hat{u_V}$ (pred)")

    plt.tight_layout()
    if save_path is not None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()     
