import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from pyDOE import lhs
import os
import json

eqnPath = "2d-ackley"
sys.path.append("utils")
from plotting import figsize
# sys.path.append(os.path.join("datagen", eqnPath))
# from names import X_FILE, U_MEAN_FILE, U_STD_FILE


def scarcify(X, u, N):
    idx = np.random.choice(X.shape[0], N, replace=False)
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[idx, :], u[idx, :], X[mask, :], u[mask, :]


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


def prep_data(n_x, n_y, n_t, x_min, x_max, y_min, y_max):
    # Number of degrees of freedom: the whole domain
    n_h = n_x * n_y
    # Since the grid is in the DOF
    nn_t = n_t

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
                U_h[j*n_x + k, i] = f_i_of_xy[j, k]
        
    # The input are directly the parameters (the space is going to be reduced by POD)
    X_U_rb = mu_lhs

    return X, Y, U_h, X_U_rb, lb, ub


def plot_results(U_h, U_h_pred=None,
                 X_U_rb_test=None, U_rb_test=None,
                 U_rb_pred=None, hp=None, save_path=None):

    dirname = os.path.join(eqnPath, "data")
    x = np.load(os.path.join(dirname, X_FILE))
    u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    u_std = np.load(os.path.join(dirname, U_STD_FILE))

    fig = plt.figure(figsize=figsize(2, 1))

    # plotting the first three coefficients u_rb
    # ax0 = fig.add_subplot(2, 2, 1)
    # For i in range(2):
    #     ax0.plot(np.sort(X_U_rb_test[:, 0]), U_rb_pred[:, i][np.argsort(X_U_rb_test[:, 0])],
    #              "b-", label=r"$\hat{u_{rb}}(\gamma_1)$")
    #     ax0.plot(np.sort(X_U_rb_test[:, 0]), U_rb_test[:, i][np.argsort(X_U_rb_test[:, 0])],
    #              "r--", label=r"$u_{rb}(\gamma_1)$")
    # ax0.legend() 
    # ax0.set_title(r"First two $U_{rb}$ coefficients")
    # ax0.set_xlabel(r"$\gamma_1$")

    # # plotting the first three coefficients u_rb
    # If X_U_rb_test.shape[1] > 1:
    #     ax00 = fig.add_subplot(2, 2, 2)
    #     for i in range(2):
    #         ax00.plot(np.sort(X_U_rb_test[:, 1]), U_rb_pred[:, i][np.argsort(X_U_rb_test[:, 1])],
    #                  "b-", label=r"$\hat{u_{rb}}(\gamma_2)$")
    #         ax00.plot(np.sort(X_U_rb_test[:, 1]), U_rb_test[:, i][np.argsort(X_U_rb_test[:, 1])],
    #                  "r--", label=r"$u_{rb}(\gamma_2)$")
    #     ax00.legend() 
    #     ax00.set_title(r"First two $U_{rb}$ coefficients")
    #     ax00.set_xlabel(r"$\gamma_2$")
        
    # plotting the means
    ax1 = fig.add_subplot(1, 2, 1)
    if U_h_pred is not None:
        ax1.plot(x, np.mean(U_h_pred, axis=1), "b-", label=r"$\hat{U_h}(x, \mu)$")
    ax1.plot(x, np.mean(U_h, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax1.plot(x, u_mean, "r,", label=r"$U_{h-lhs}(x, \mu)$")
    ax1.legend()
    ax1.set_title("Means")

    ax2 = fig.add_subplot(1, 2, 2)
    if U_h_pred is not None:
        ax2.plot(x, np.std(U_h_pred, axis=1), "b-", label=r"$\hat{U_h}(x, \mu)$")
    ax2.plot(x, np.std(U_h, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax2.plot(x, u_std, "r,", label=r"$U_{h-lhs}(x, \mu)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    
    if save_path != None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()
