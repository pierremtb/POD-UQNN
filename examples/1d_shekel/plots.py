"""Module for plotting results of 1D Shekel Equation."""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.plotting import figsize, saveresultdir
from podnn.metrics import error_podnn
from podnn.testgenerator import X_FILE, U_MEAN_FILE, U_STD_FILE


def get_test_data():
    dirname = "data"
    X = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, U_test_mean[0, :], U_test_std[0, :]


def plot_results(U, U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                 train_res, HP=None, no_plot=False):

    X, U_test_mean, U_test_std = get_test_data()
    x = X[0]

    U_pred_mean = np.mean(U_pred, axis=1)
    U_pred_std = np.std(U_pred, axis=1)
    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    hifi_error_test_mean = 100 * error_podnn(U_test_mean, U_pred_hifi_mean)
    hifi_error_test_std = 100 * error_podnn(U_test_std, U_pred_hifi_std)
    print("--")
    print(f"Error on the mean test HiFi LHS solution: {error_test_mean:4f}%")
    print(f"Error on the stdd test HiFi LHS solution: {error_test_std:4f}%")
    print("--")
    print(f"Hifi Error on the mean test HiFi LHS solution: {hifi_error_test_mean:4f}%")
    print(f"Hifi Error on the stdd test HiFi LHS solution: {hifi_error_test_std:4f}%")
    print("--")

    if no_plot:
        return error_test_mean, error_test_std

    fig = plt.figure(figsize=figsize(1, 2, 2))

    # Plotting the means
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, np.mean(U_pred, axis=1), "b-", label=r"$\hat{u_V}(x)$")
    ax1.plot(x, np.mean(U, axis=1), "r--", label=r"$u_V(x)$")
    ax1.plot(x, U_test_mean, "k,", label=r"$u_T(x)$")
    ax1.plot(x, U_pred_hifi_mean, "b,", label=r"$\hat{u_T}(x)$")
    ax1.legend()
    ax1.set_title("Means")
    ax1.set_xlabel("$x$")

    # Plotting the std
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, np.std(U_pred, axis=1), "b-", label=r"$\hat{u_V}(x)$")
    ax2.plot(x, np.std(U, axis=1), "r--", label=r"$u_V(x)$")
    ax2.plot(x, U_test_std, "k,", label=r"$u_T(x)$")
    ax2.plot(x, U_pred_hifi_std, "b,", label=r"$\hat{u_T}(x)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    ax2.set_xlabel("$x$")

    saveresultdir(HP, train_res)

    return error_test_mean, error_test_std


if __name__ == "__main__":
    from hyperparams import HP as hp

    model = PodnnModel.load("cache")

    x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
    _, _, X_v_test, _, U_val = model.load_train_data()

    # Predict and restruct
    U_pred = model.predict(X_v_test)

    # Sample the new model to generate a HiFi prediction
    X_v_test_hifi = model.generate_hifi_inputs(hp["n_s_hifi"], hp["mu_min"], hp["mu_max"])
    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)
    U_pred_hifi_mean = U_pred_hifi_mean.reshape((hp["n_x"],))
    U_pred_hifi_std = U_pred_hifi_std.reshape((hp["n_x"],))

    # Plot and save the results
    plot_results(U_val, U_pred, U_pred_hifi_mean, U_pred_hifi_std, hp)
