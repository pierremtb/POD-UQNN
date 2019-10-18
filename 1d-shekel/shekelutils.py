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
from metrics import error_podnn
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, U_MEAN_FILE, U_STD_FILE


def prep_data(n_x, n_s, bet_count=0, gam_count=3):
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
    X_v = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Creating the snapshots
    print(f"Generating {n_s} corresponding snapshots")
    U = np.zeros((n_x, n_s))
    x = np.linspace(0, 10, n_x)
    for i in tqdm(range(n_s)):
        # Altering the beta params with lhs perturbations
        bet_kxsi = X_v[i, :bet_count]
        bet[0:bet_kxsi.shape[0], 0] = bet_kxsi
        # Altering the gamma params with lhs perturbations
        gam_kxsi = X_v[i, bet_count:]
        gam[0:gam_kxsi.shape[0], 0] = gam_kxsi

        # Calling the Shekel function
        U[:, i] = -shekel(x[None, :], gam, bet)[0]

    return U, X_v, lb, ub


def get_test_data():
    dirname = os.path.join(eqnPath, "data")
    x = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return x, U_test_mean, U_test_std


def plot_results(U, U_pred=None,
                 hp=None, save_path=None):

    x, U_test_mean, U_test_std = get_test_data()

    U_pred_mean = np.mean(U_pred, axis=1)
    U_pred_std = np.std(U_pred, axis=1)
    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    if save_path is not None:
        print("--")
        print(f"Error on the mean test HiFi LHS solution: {error_test_mean:4f}%")
        print(f"Error on the stdd test HiFi LHS solution: {error_test_std:4f}%")
        print("--")

    fig = plt.figure(figsize=figsize(1, 2, 2))

    # Plotting the means
    ax1 = fig.add_subplot(1, 2, 1)
    if U_pred is not None:
        ax1.plot(x, np.mean(U_pred, axis=1), "b-", label=r"$\hat{u_V}(x)$")
    ax1.plot(x, np.mean(U, axis=1), "r--", label=r"$u_V(x)$")
    ax1.plot(x, U_test_mean, "r,", label=r"$u_T(x)$")
    ax1.legend()
    ax1.set_title("Means")
    ax1.set_xlabel("$x$")

    # Plotting the std
    ax2 = fig.add_subplot(1, 2, 2)
    if U_pred is not None:
        ax2.plot(x, np.std(U_pred, axis=1), "b-", label=r"$\hat{u_V}(x)$")
    ax2.plot(x, np.std(U, axis=1), "r--", label=r"$u_V(x)$")
    ax2.plot(x, U_test_std, "r,", label=r"$u_T(x)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    ax2.set_xlabel("$x$")
    
    if save_path is not None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()

    # Table display of the errors
    # ax = fig.add_subplot(2, 2, 3)
    # ax.axis('off')
    # table = r"\textbf{Numerical results}  \\ \\ " + \
    #         r"\begin{tabular}{|l|c|} " + \
    #         r"\hline " + \
    #         r"Validation error (%.1f\%% of the dataset) & $%.4f \%%$ \\ " % (100 * hp["train_val_ratio"], error_val) + \
    #         r"\hline " + \
    #         r"Test error (HiFi LHS sampling) & $%.4f \%%$ \\ " % (error_test_mean) + \
    #         r"\hline " + \
    #         r"\end{tabular}"
    # ax.text(0.1, 0.1, table)
