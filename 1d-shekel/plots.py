import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from pyDOE import lhs
from deap.benchmarks import shekel
import json

EQN_PATH = "1d-shekel"
sys.path.append("utils")
from plotting import figsize, saveresultdir, savefig
from metrics import error_podnn
from testgenerator import X_FILE, U_MEAN_FILE, U_STD_FILE


def get_test_data():
    dirname = os.path.join(EQN_PATH, "data")
    X = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, U_test_mean, U_test_std


def plot_results(U, U_pred=None,
                 HP=None, save_path=None, no_plot=False):

    X, U_test_mean, U_test_std = get_test_data()
    x = X[0]

    U_pred_mean = np.mean(U_pred, axis=1)
    U_pred_std = np.std(U_pred, axis=1)
    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    if save_path is not None:
        print("--")
        print(f"Error on the mean test HiFi LHS solution: {error_test_mean:4f}%")
        print(f"Error on the stdd test HiFi LHS solution: {error_test_std:4f}%")
        print("--")

        if no_plot:
            return error_test_mean, error_test_std

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
        saveresultdir(save_path, save_HP=HP)
    else:
        plt.show()

    return error_test_mean, error_test_std
