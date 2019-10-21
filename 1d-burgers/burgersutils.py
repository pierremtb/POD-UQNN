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
from plotting import figsize, saveresultdir, savefig
from metrics import error_podnn
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, U_MEAN_FILE, U_STD_FILE


def prep_data(n_x, x_min, x_max, n_t, t_min, t_max, n_s, mu_mean):
    # Total number of snapshots
    nn_s = n_t*n_s

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling...")
    pbar = tqdm(total=100)
    X = lhs(n_s, 1).T
    pbar.update(50)
    lb = mu_mean * (1 - np.sqrt(3))
    ub = mu_mean * (1 + np.sqrt(3))
    mu_lhs = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()

    # Number of inputs in time plus number of parameters
    n_d = 1 + p_var.shape[0]
    
    # Creating the snapshots
    print(f"Generating {nn_s} corresponding snapshots")
    U = np.zeros((n_x, n_s))
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    for i in tqdm(range(n_s)):
        # Calling the Shekel function
        U[:, i] = -shekel(x, t, mu_lhs[i, :])

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
