"""Module for plotting results of 1D Shekel Equation."""

import os
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.plotting import figsize, saveresultdir, savefig
from podnn.metrics import re, re_mean_std
from podnn.testgenerator import X_FILE, U_MEAN_FILE, U_STD_FILE


def get_test_data():
    dirname = "data"
    X = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, U_test_mean, U_test_std


def plot_results(U_test, U_pred, U_pred_hifi_mean, U_pred_hifi_std, sigma_pod,
                 resdir=None, train_res=None, HP=None, no_plot=False):

    X, U_test_hifi_mean, U_test_hifi_std = get_test_data()
    x = X[0]

    U_pred_mean = np.mean(U_pred, axis=-1)
    # Using nanstd() to prevent NotANumbers from appearing
    U_pred_std = np.nanstd(U_pred, axis=-1)

    U_pred_hifi_mean_sig = U_pred_hifi_mean[1]
    U_pred_hifi_std_sig = U_pred_hifi_std[1]

    U_pred_hifi_mean = U_pred_hifi_mean[0]
    U_pred_hifi_std = U_pred_hifi_std[0]

    # Compute relative error
    error_test_mean, error_test_std = re_mean_std(U_test, U_pred)
    hifi_error_test_mean = re(U_test_hifi_mean, U_pred_hifi_mean)
    hifi_error_test_std = re(U_test_hifi_std, U_pred_hifi_std)
    sigma_Thf = U_pred_hifi_mean_sig.mean(0).mean(0)
    print(f"Test relative error: mean {error_test_mean:4f}, std {error_test_std:4f}")
    print(f"HiFi test relative error: mean {hifi_error_test_mean:4f}, std {hifi_error_test_std:4f}")
    print(f"Mean Sigma on hifi predictions: {sigma_Thf:4f}")
    print(f"Mean Sigma contrib from POD: {sigma_pod:4f}")
    errors = {
        "REM_T": error_test_mean.item(),
        "RES_T": error_test_std.item(),
        "REM_Thf": hifi_error_test_mean.item(),
        "RES_Thf": hifi_error_test_std.item(),
        "sigma": sigma_Thf.item(),
        "sigma_pod": sigma_pod.item(),
    }

    if no_plot:
        return hifi_error_test_mean, hifi_error_test_std

    fig = plt.figure(figsize=figsize(1, 2, scale=2.))

    # Plotting the means
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, U_pred_mean[0], "k,", label=r"$\hat{u}_D(x)$")
    ax1.plot(x, U_pred_hifi_mean[0], "b-", label=r"$\hat{u}_{D,tst}(x)$")
    ax1.plot(x, U_test_hifi_mean[0], "r--", label=r"$u_{D,tst}(x)$")
    lower = U_pred_hifi_mean[0] - 2 * U_pred_hifi_mean_sig[0]
    upper = U_pred_hifi_mean[0] + 2 * U_pred_hifi_mean_sig[0]
    plt.fill_between(x, lower, upper, 
                     facecolor='C0', alpha=0.3, label=r"$2\sigma_{D,hf}(x)$")
    ax1.legend()
    ax1.set_title("Means")
    ax1.set_xlabel("$x$")

    # Plotting the std
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, U_pred_std[0], "k,", label=r"$\hat{u}_T(x)$")
    ax2.plot(x, U_pred_hifi_std[0], "b-", label=r"$\hat{u}_{T,hf}(x)$")
    ax2.plot(x, U_test_hifi_std[0], "r--", label=r"$u_{T,hf}(x)$")
    lower = U_pred_hifi_std[0] - 2 * U_pred_hifi_std_sig[0]
    upper = U_pred_hifi_std[0] + 2 * U_pred_hifi_std_sig[0]
    plt.fill_between(x, lower, upper, 
                     facecolor='C0', alpha=0.3, label=r"2\text{std}(\hat{u}_T(x))")
    ax2.set_title("Standard deviations")
    ax2.set_xlabel("$x$")

    if resdir is None:
        return hifi_error_test_mean, hifi_error_test_std
        
    saveresultdir(resdir, HP, errors, train_res)

    return hifi_error_test_mean, hifi_error_test_std


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise FileNotFoundError("Provide a resdir")

    resdir = sys.argv[1]
    with open(os.path.join(resdir, "HP.txt")) as HPFile:
        hp = yaml.load(HPFile)

    model = PodnnModel.load(resdir)

    x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))
    _, _, _, X_v_test, U_test = model.load_train_data()

    U_pred, U_pred_sig = model.predict(X_v_test)
    U_pred = model.restruct(U_pred)
    U_pred_sig = model.restruct(U_pred_sig)
    U_test = model.restruct(U_test)

    # Sample the new model to generate a HiFi prediction
    print("Sampling {n_s_hifi} parameters")
    X_v_test_hifi = model.generate_hifi_inputs(hp["n_s_hifi"],
                                               hp["mu_min"], hp["mu_max"])
    print("Predicting the {n_s_hifi} corresponding solutions")
    U_pred_hifi, U_pred_hifi_sig = model.predict(X_v_test_hifi)
    U_pred_hifi_mean = (model.restruct(U_pred_hifi.mean(-1), no_s=True),
                        model.restruct(U_pred_hifi_sig.mean(-1), no_s=True))
    U_pred_hifi_std = (model.restruct(U_pred_hifi.std(-1), no_s=True),
                       model.restruct(U_pred_hifi_sig.mean(-1), no_s=True))

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling on the non-spatial params...")
    from genhifi import u
    # mu_min_out, mu_min = np.array(hp["mu_min_out"]), np.array(hp["mu_min"])
    # mu_lhs_out_min = model.sample_mu(100, mu_min_out, mu_min)
    # n_d = mu_lhs_out_min.shape[1]
    # n_h = hp["n_v"] * x_mesh.shape[0]
    # X_v_test_out, U_test_out, U_test_out_struct, _ = \
    #     model.create_snapshots(n_d, n_h, u, mu_lhs_out_min)
    # Projecting
    # v_pred_out, v_pred_out_sig = model.predict_v(mu_lhs_out_min)

    # mu_max, mu_max_out = np.array(hp["mu_max"]), np.array(hp["mu_max_out"])
    # mu_lhs_out_max = model.sample_mu(100, mu_max, mu_max_out)
    # X_v_test_out_max, U_test_out_max, U_test_out_struct_max, _ = \
    #     model.create_snapshots(n_d, n_h, u, mu_lhs_out_max)
    # # Projecting
    # v_pred_out_max, v_pred_out_sig_max = model.predict_v(X_v_test_out_max)
    # after = v_pred_out_sig_max.mean(-1)

    # _, v_pred_sig = model.predict_v(X_v_test)
    # _, U_pred_out_sig = model.predict(X_v_test)
    # plt.plot(mu_lhs_out_min[:, 0], U_pred_out_sig.mean(0), "bo")
    # plt.plot(X_v_test[:, 0], U_pred_sig[0].mean(0), "ro")
    # plt.plot(mu_lhs_out_min[:, 0], v_pred_out_sig.mean(-1), "bo")
    # plt.plot(X_v_test[:, 0], v_pred_sig.mean(-1), "ro")
    # plt.plot(mu_lhs_out_max[:, 0], after, "bo")
    # plt.show()


    # Plot against test and save
    plot_results(U_test, U_pred, U_pred_hifi_mean, U_pred_hifi_std, model.pod_sig.mean(),
                        resdir, train_res=None, HP=hp)

    # Add samples graph
    n_samples = 3
    mu_lhs_in = model.sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
    mu_lhs_out = model.sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))

    n_plot_x = 2
    n_plot_y = n_samples
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)
    for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
        X_v_samples, U_samples, _, _ = \
            model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
                                
        x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
        idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
        for col, idx_i in enumerate(idx):
            lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
            X_i = X_v_samples[idx_i, :].reshape(1, -1)
            U_pred_i, U_pred_i_var = model.predict(X_i)
            ax = fig.add_subplot(gs[row, col])
            ax.plot(x, U_pred_i, "C0-", label=r"$u_D(s_{" + lbl + r"})$")
            ax.plot(x, U_samples[:, idx_i], "r--", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
            lower = U_pred_i[:, 0] - 3*U_pred_i_var[:, 0]
            upper = U_pred_i[:, 0] + 3*U_pred_i_var[:, 0]
            ax.fill_between(x, lower, upper, label=r"$3\sigma_D(s_{" + lbl + r"})$",
                            alpha=0.2, facecolor="C0")
            ax.set_xlabel("$x$")
            if col == len(idx) - 1:
                ax.legend()
    savefig(os.path.join(resdir, "graph-samples"))

