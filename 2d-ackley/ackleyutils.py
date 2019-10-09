import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from pyDOE import lhs
import os
import json
from scipy.interpolate import griddata

eqnPath = "2d-ackley"
sys.path.append("utils")
from plotting import figsize, saveresultdir
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, Y_FILE, U_MEAN_FILE, U_STD_FILE


def scarcify(X, u, N):
    idx = np.random.choice(X.shape[0], N, replace=False)
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[idx, :], u[idx, :], X[mask, :], u[mask, :]


def restruct(U_h, n_x, n_y, n_s):
    U_h_struct = np.zeros((n_x, n_y, n_s))
    idx = np.arange(n_s) * n_x * n_y
    for i in range(n_x):
        U_h_struct[i, :] = U_h[:, idx + i]
    return U_h_struct


# The custom stochastic Ackley 2D function
def u_h(x, y, mu):
    return - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
           - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
           + 20 + np.exp(1)


def prep_data(n_x, n_y, n_s, x_min, x_max, y_min, y_max):
    # Number of degrees of freedom: the whole domain
    n_h = n_x * n_y
    # Since the grid is in the DOF
    nn_s = n_s

    # Auckley params means
    mu = np.array([0., 0., 0.])

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling")
    pbar = tqdm(total=100)
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
    print(f"Generating {nn_s} corresponding snapshots")
    X_U_rb = np.zeros((nn_s, n_d))
    U_h = np.zeros((n_h, nn_s))
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(x_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)
    for i in tqdm(range(n_s)):
        mu_i = mu_lhs[i, :]

        # Calling the Ackley function
        f_i_of_xy = u_h(X, Y, mu_i)
        U_h[:, i] = np.reshape(f_i_of_xy, (n_x * n_y,))
        # for j, x_j in enumerate(x):
        #     for k, y_k in enumerate(y):
        #         U_h[j*n_x + k, i] = f_i_of_xy[j, k]
        
    # The input are directly the parameters (the space is going to be reduced by POD)
    X_U_rb = mu_lhs

    return X, Y, U_h, X_U_rb, lb, ub


def plot_results(U_h, U_h_pred=None,
                 hp=None, save_path=None):

    dirname = os.path.join(eqnPath, "data")
    X = np.load(os.path.join(dirname, X_FILE))
    x = X[0, :]
    Y = np.load(os.path.join(dirname, Y_FILE))
    y = Y[0, :]
    u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    u_std = np.load(os.path.join(dirname, U_STD_FILE))

    levels = list(range(2, 15))
    fig = plt.figure(figsize=figsize(2, 1))

    # plotting the means
    ax1 = fig.add_subplot(221)
    ct1 = ax1.contourf(X, Y, np.mean(U_h, axis=2), levels=levels, origin="lower")
    plt.colorbar(ct1)
    ax1.set_title("Mean of $U_h$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    if U_h_pred is not None:
        ax2 = fig.add_subplot(222)
        ct2 = ax2.contourf(X, Y, np.mean(U_h_pred, axis=2), levels=levels)
        plt.colorbar(ct2)
        ax2.set_title("Mean of $\hat{U_h}$")
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")

    ax3 = fig.add_subplot(223)
    ax3.contourf(X, Y, np.std(U_h, axis=2))
    ax3.set_title("Standard deviation of $U_h$")
    if U_h_pred is not None:
        ax4 = fig.add_subplot(224)
        ax4.contourf(X, Y, np.std(U_h_pred, axis=2))
        ax4.set_title("Standard deviation of $\hat{U_h}$")
    
    if save_path != None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()
