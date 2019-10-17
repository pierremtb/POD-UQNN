import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
from tqdm import tqdm
import tensorflow as tf
from pyDOE import lhs
import os
import json
from scipy.interpolate import griddata

eqnPath = "2d-ackley"
sys.path.append("utils")
from plotting import figsize, saveresultdir
from pod import get_pod_bases
from metrics import error_podnn
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, Y_FILE, U_MEAN_FILE, U_STD_FILE


# The custom stochastic Ackley 2D function
def u_h(x, y, mu):
    return - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
           - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
           + 20 + np.exp(1)


def prep_data(n_x, n_y, n_s, x_min, x_max, y_min, y_max, disable_progress=False):
    # Number of degrees of freedom: the whole domain
    n_h = n_x * n_y
    # Since the grid is in the DOF
    nn_s = n_s

    # Auckley params means
    mu = np.array([0., 0., 0.])

    # LHS sampling (first uniform, then perturbated)
    if not disable_progress:
        print("Doing the LHS sampling...")
    pbar = tqdm(total=100, disable=disable_progress)
    X = lhs(n_s, mu.shape[0]).T
    pbar.update(50)
    lb = -1. * np.ones_like(mu)
    ub = +1. * np.ones_like(mu)
    mu_lhs = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Number of inputs in space plus number of parameters
    n_d = 2 + mu.shape[0]

    # Creating the snapshots
    if not disable_progress:
        print(f"Generating {nn_s} corresponding snapshots...")
    X_U_rb = np.zeros((nn_s, n_d))
    U_h = np.zeros((n_h, nn_s))
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(x_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)
    for i in tqdm(range(n_s), disable=disable_progress):
        U_h[:, i] = np.reshape(u_h(X, Y, mu_lhs[i, :]), (n_x * n_y,))
        
    # The input are directly the parameters (the space is going to be reduced by POD)
    X_U_rb = mu_lhs

    return X, Y, U_h, X_U_rb, lb, ub


def plot_contour(fig, pos, X, Y, U, levels, title):
    ax = fig.add_subplot(pos)
    ct = ax.contourf(X, Y, U, levels=levels, origin="lower")
    plt.colorbar(ct)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def get_test_data():
    dirname = os.path.join(eqnPath, "data")
    X = np.load(os.path.join(dirname, X_FILE))
    Y = np.load(os.path.join(dirname, Y_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, Y, U_test_mean, U_test_std


def plot_results(U_h, U_h_pred=None,
                 hp=None, save_path=None):
    X, Y, U_test_mean, U_test_std = get_test_data()

    U_pred_mean = np.mean(U_h_pred, axis=2)
    U_pred_std = np.std(U_h_pred, axis=2)
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
                 X, Y, U_test_mean,
                 mean_levels, "Mean of $u_T$ (test)")
    plot_contour(fig, gs[0:2, 2:4],
                 X, Y, np.mean(U_h, axis=2),
                 mean_levels, "Mean of $u_V$ (val)")
    if U_h_pred is not None:
        plot_contour(fig, gs[0:2, 4:6],
                     X, Y, np.mean(U_h_pred, axis=2),
                     mean_levels, "Mean of $\hat{u_V}$ (pred)")
    plot_contour(fig, gs[2:4, 0:2],
                 X, Y, U_test_std,
                 std_levels, "Standard deviation of $u_T$ (test)")
    plot_contour(fig, gs[2:4, 2:4],
                 X, Y, np.std(U_h, axis=2),
                 std_levels, "Standard deviation of $u_V$ (val)")
    if U_h_pred is not None:
        plot_contour(fig, gs[2:4, 4:6],
                     X, Y, np.std(U_h_pred, axis=2),
                     std_levels, "Standard deviation of $\hat{u_V}$ (pred)")

    plt.tight_layout()
    if save_path is not None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()     

    # Table display of the errors
    # ax = fig.add_subplot(gs[4, :])
    # ax.axis('off')
    # error_val = 100 * error_podnn(U_h, U_h_pred)
    # table = r"\textbf{Numerical results}  \\ \\ " + \
    #         r"\begin{tabular}{|l|c|} " + \
    #         r"\hline " + \
    #         r"Validation error (%.1f\%% of the dataset) & $%.4f \%%$ \\ " % (100 * hp["train_val_ratio"], error_val) + \
    #         r"\hline " + \
    #         r"Test error (HiFi LHS sampling) & $%.4f \%%$ \\ " % (error_test_mean) + \
    #         r"\hline " + \
    #         r"\end{tabular}"
    # ax.text(0.1, 0.1, table)
