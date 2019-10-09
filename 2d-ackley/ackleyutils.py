import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from pyDOE import lhs
import json

eqnPath = "2d-ackley"
sys.path.append("utils")
from plotting import figsize, saveresultdir
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, Y_FILE, U_MEAN_FILE, U_STD_FILE


def restruct(U_h, n_x, n_y, n_t):
    U_h_struct = np.zeros((n_x, n_y, n_t))
    idx = np.arange(n_t) * n_x * n_y
    for i in range(n_x):
        U_h_struct[i, :] = U_h[:, idx + i]
    return U_h_struct

# The custom stochastic Ackley 2D function
def u_h(x, y, mu):
    return - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
           - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
           + 20 + np.exp(1)


def prep_data(n_h, n_x, n_y, n_t, x_min, x_max, y_min, y_max):
    # Total number of snapshots
    nn_t = n_x*n_y*n_t

    # Auckley params mean
    mu = np.array([0., 0., 0.])

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling")
    pbar = tqdm(total=100)
    X = lhs(n_t, mu.shape[0]).T
    pbar.update(50)
    lb = -1. * np.ones_like(mu)
    ub = +1. * np.ones_like(mu)
    mu_lhs = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Number of inputs in space plus number of parameters
    n_d = 2 + mu.shape[0]

    # Creating the snapshots
    print(f"Generating {nn_t} corresponding snapshots")
    X_U_rb = np.zeros((nn_t, n_d))
    U_h = np.zeros((n_h, nn_t))
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(x_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)
    for i in tqdm(range(n_t)):
        mu_i = mu_lhs[i, :]

        # Calling the Ackley function
        f_i_of_xy = u_h(X, Y, mu_i)
        for j, x_j in enumerate(x):
            for k, y_k in enumerate(y):
                U_h[:, i*n_x*n_y + j*n_x + k] = f_i_of_xy[j, k]
                X_U_rb[i*n_x*n_y + j*n_x + k, :] = np.hstack(([x_j, y_k], mu_lhs[i, :]))
        
    lb = np.hstack(([x_min, y_min], lb))
    ub = np.hstack(([x_max, y_max], ub))

    return X, Y, U_h, X_U_rb, lb, ub


def plot_results(U_h_train, U_h_pred=None,
                 X_U_rb_test=None, U_rb_test=None,
                 U_rb_pred=None, hp=None, save_path=None):

    dirname = os.path.join(eqnPath, "data")
    X = np.load(os.path.join(dirname, X_FILE))
    Y = np.load(os.path.join(dirname, Y_FILE))
    u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    u_std = np.load(os.path.join(dirname, U_STD_FILE))

    fig = plt.figure(figsize=figsize(2, 1))

    # plotting the means
    ax1 = fig.add_subplot(121, projection="3d")
    if U_h_pred is not None:
        ax1.plot_surface(X, Y, np.mean(U_h_pred, axis=1), "b-", label=r"$\hat{U_h}(x, \mu)$")
    ax1.plot_surface(X, Y, np.mean(U_h_train, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax1.plot_surface(X, Y, u_mean, "r,", label=r"$U_{h-lhs}(x, \mu)$")
    ax1.legend()
    ax1.set_title("Means")

    ax2 = fig.add_subplot(122, projection="3d")
    if U_h_pred is not None:
        ax2.plot_surface(X, Y, np.std(U_h_pred, axis=1), "b-", label=r"$\hat{U_h}(x, \mu)$")
    ax2.plot_surface(X, Y, np.std(U_h_train, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax2.plot_surface(X, Y, u_std, "r,", label=r"$U_{h-lhs}(x, \mu)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    
    if save_path != None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()
