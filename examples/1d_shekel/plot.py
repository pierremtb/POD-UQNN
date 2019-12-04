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
    return X, U_test_mean, U_test_std


def plot_results(U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                 train_res=None, HP=None, no_plot=False):

    X, U_test_hifi_mean, U_test_hifi_std = get_test_data()
    x = X[0]

    U_pred_mean = np.mean(U_pred, axis=-1)
    # Using nanstd() to prevent NotANumbers from appearing
    U_pred_std = np.nanstd(U_pred, axis=-1)

    hifi_error_test_mean = error_podnn(U_test_hifi_mean, U_pred_hifi_mean)
    hifi_error_test_std = error_podnn(U_test_hifi_std, U_pred_hifi_std)
    print(f"HiFi test relative error: mean {hifi_error_test_mean:4f}, std {hifi_error_test_std:4f}")

    if no_plot:
        return hifi_error_test_mean, hifi_error_test_std

    fig = plt.figure(figsize=figsize(1, 2, 2))

    # Plotting the means
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, U_pred_mean[0], "k,", label=r"$\hat{u}_T(x)$")
    ax1.plot(x, U_pred_hifi_mean[0], "b-", label=r"$\hat{u}_T^{hf}(x)$")
    ax1.plot(x, U_test_hifi_mean[0], "r--", label=r"$u_T^{hf}(x)$")
    ax1.legend()
    ax1.set_title("Means")
    ax1.set_xlabel("$x$")

    # Plotting the std
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, U_pred_std[0], "k,", label=r"$\hat{u}_T(x)$")
    ax2.plot(x, U_pred_hifi_std[0], "b-", label=r"$\hat{u}_T^{hf}(x)$")
    ax2.plot(x, U_test_hifi_std[0], "r--", label=r"$u_T^{hf}(x)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    ax2.set_xlabel("$x$")

    saveresultdir(HP, train_res)

    return hifi_error_test_mean, hifi_error_test_std


if __name__ == "__main__":
    from hyperparams import HP as hp

    model = PodnnModel.load("cache")

    x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
    _, _, X_v_test, _, U_test = model.load_train_data()

    # Predict and restruct
    U_pred = model.predict(X_v_test)
    U_pred = model.restruct(U_pred)
    U_test = model.restruct(U_test)

    # Sample the new model to generate a HiFi prediction
    X_v_test_hifi = model.generate_hifi_inputs(hp["n_s_hifi"], hp["mu_min"], hp["mu_max"])
    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)

    # Plot and save the results
    plot_results(U_pred, U_pred_hifi_mean, U_pred_hifi_std, HP=hp)
