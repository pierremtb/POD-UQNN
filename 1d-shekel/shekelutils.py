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
from plotting import figsize, saveresultdir, savefig
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, U_MEAN_FILE, U_STD_FILE

def scarcify(X, u, N):
    idx = np.random.choice(X.shape[0], N, replace=False)
    # return X[idx, :], u[idx, :], 
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[idx, :], u[idx, :], X[mask, :], u[mask, :] 


def prep_data(n_e, n_s, bet_count=0, gam_count=3):
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
    X_U_rb = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Creating the snapshots
    print(f"Generating {n_s} corresponding snapshots")
    U_h = np.zeros((n_e, n_s))
    x = np.linspace(0, 10, n_e)
    for i in tqdm(range(n_s)):
        # Altering the beta params with lhs perturbations
        bet_kxsi = X_U_rb[i, :bet_count]
        bet[0:bet_kxsi.shape[0], 0] = bet_kxsi
        # Altering the gamma params with lhs perturbations
        gam_kxsi = X_U_rb[i, bet_count:]
        gam[0:gam_kxsi.shape[0], 0] = gam_kxsi

        # Calling the Shekel function
        U_h[:, i] = -shekel(x[None, :], gam, bet)[0]

    return U_h, X_U_rb, lb, ub


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
