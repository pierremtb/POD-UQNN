import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from pyDOE import lhs
from deap.benchmarks import shekel
import json

eqnPath = "1d-shekel"
sys.path.append("utils")
from plotting import figsize, saveresultsdir, savefig 
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, U_MEAN_FILE, U_STD_FILE


def restruct(U_h, n_x, n_s):
    U_h_struct = np.zeros((n_x, n_s))
    idx = np.arange(n_s) * n_x
    for i in range(n_x):
        U_h_struct[i, :] = U_h[:, idx + i]
    return U_h_struct


def prep_data(n_h, n_x, n_s, bet_count=0, gam_count=3):
    # Total number of snapshots
    nn_s = n_x*n_s

    x_min = 0.
    x_max = 10.

    # Shekel parameters (t=10-sized)
    bet = 1/10 * np.array([[1, 2, 2, 4, 4, 6, 3, 7, 5, 5]]).T
    gam = 1. * np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7]]).T
    mu = (bet, gam)
    
    # Perturbations
    bet_var = np.hstack((bet, 0.1 * bet))[:bet_count, :]
    gam_var = np.hstack((gam, 0.1 * gam))[:gam_count, :]
    p_var = np.vstack((bet_var, gam_var))
    
    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling")
    pbar = tqdm(total=100)
    X = lhs(n_s, p_var.shape[0]).T
    pbar.update(50)
    lb = p_var[:, 0] - np.sqrt(3)*p_var[:, 1]
    ub = p_var[:, 0] + np.sqrt(3)*p_var[:, 1]
    mu_lhs = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Number of inputs in space plus number of parameters
    n_d = 1 + p_var.shape[0]

    # Creating the snapshots
    print(f"Generating {nn_s} corresponding snapshots")
    X_U_rb = np.zeros((nn_s, n_d))
    U_h = np.zeros((n_h, nn_s))
    x = np.linspace(x_min, x_max, n_x)
    for i in tqdm(range(n_s)):
        # Altering the beta params with lhs perturbations
        bet_kxsi = mu_lhs[i, :bet_count]
        bet[0:bet_kxsi.shape[0], 0] = bet_kxsi
        # Altering the gamma params with lhs perturbations
        gam_kxsi = mu_lhs[i, bet_count:]
        gam[0:gam_kxsi.shape[0], 0] = gam_kxsi

        # Calling the Shekel function
        f_i_of_x = -shekel(x[None, :], gam, bet)[0]
        for j, x_j in enumerate(x):
            U_h[:, i*n_x + j] = f_i_of_x[j]
            X_U_rb[i*n_x + j, :] = np.hstack(([x_j], mu_lhs[i, :]))
    
    lb = np.hstack(([x_min], lb))
    ub = np.hstack(([x_max], ub))

    return U_h, X_U_rb, lb, ub


def plot_results(U_h_train, U_h_pred=None,
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
    ax1.plot(x, np.mean(U_h_train, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax1.plot(x, u_mean, "r,", label=r"$U_{h-lhs}(x, \mu)$")
    ax1.legend()
    ax1.set_title("Means")

    ax2 = fig.add_subplot(1, 2, 2)
    if U_h_pred is not None:
        ax2.plot(x, np.std(U_h_pred, axis=1), "b-", label=r"$\hat{U_h}(x, \mu)$")
    ax2.plot(x, np.std(U_h_train, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax2.plot(x, u_std, "r,", label=r"$U_{h-lhs}(x, \mu)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    
    if save_path != None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()
