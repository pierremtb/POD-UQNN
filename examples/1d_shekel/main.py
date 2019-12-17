"""POD-NN modeling for 1D Shekel Equation."""

import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_mean_std
from podnn.mesh import create_linear_mesh

from genhifi import u, generate_test_dataset
from plot import plot_results


def main(hp, gen_test=False, use_cached_dataset=False,
         no_plot=False):
    """Full example to run POD-NN on 1d_shekel."""

    if gen_test:
        generate_test_dataset()

    if not use_cached_dataset:
        # Create linear space mesh
        x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
        np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
    else:
        x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))

    # Init the model
    model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])

    # Generate the dataset from the mesh and params
    X_v_train, v_train, \
        X_v_test, v_test, \
        U_test = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                        hp["n_s"],
                                        hp["train_val_test"],
                                        hp["eps"],
                                        u_noise=0.05,
                                        use_cache=use_cached_dataset)

    # Train
    model.initNN(hp["h_layers"], hp["h_layers_t"],
                 hp["lr"], hp["lambda"], hp["beta"],
                 hp["k1"], hp["k2"], hp["norm"])
    train_res = model.train(X_v_train, v_train, hp["epochs"],
                            hp["train_val_test"], freq=hp["log_frequency"])

    # Predict and restruct
    v_pred, v_pred_sig = model.predict_v(X_v_test)
    print(X_v_test[0, :], v_pred[0, :], v_test[0, :])
    x = np.arange(1, v_pred.shape[1] + 1)
    plt.plot(x, v_pred[0, :], "b-")
    plt.plot(x, v_test[0, :], "r--")
    lower = v_pred[0, :] - 2.0*v_pred_sig[0, :]
    upper = v_pred[0, :] + 2.0*v_pred_sig[0, :]
    plt.fill_between(x, lower, upper, 
                        facecolor='orange', alpha=0.5, label="Two std band")
    print(X_v_test[50, :], v_pred[50, :], v_test[50, :])
    plt.plot(x, v_pred[50, :], "c-")
    plt.plot(x, v_test[50, :], "k--")
    lower = v_pred[50, :] - 2.0*v_pred_sig[50, :]
    upper = v_pred[50, :] + 2.0*v_pred_sig[50, :]
    plt.fill_between(x, lower, upper, 
                        facecolor='orange', alpha=0.5, label="Two std band")
    plt.show()

    U_pred = model.V.dot(v_pred.T)
    Sigma_pred = model.V.dot(v_pred_sig.T)
    # print(U_pred.shape, Sigma_pred.shape)

    # x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    # plt.plot(x, U_pred[:, 0], "b-")
    # plt.plot(x, U_test[:, 0], "r--")
    # lower = U_pred[:, 0] - 2.0*Sigma_pred[:, 0]
    # upper = U_pred[:, 0] + 2.0*Sigma_pred[:, 0]
    # plt.fill_between(x, lower, upper, 
    #                     facecolor='orange', alpha=0.5, label="Two std band")
    # plt.show()

    U_pred = model.restruct(U_pred)
    U_test = model.restruct(U_test)

    # Compute relative error
    error_test_mean, error_test_std = re_mean_std(U_test, U_pred)
    print(f"Test relative error: mean {error_test_mean:4f}, std {error_test_std:4f}")

    # Sample the new model to generate a HiFi prediction
    print("Sampling {n_s_hifi} parameters")
    X_v_test_hifi = model.generate_hifi_inputs(hp["n_s_hifi"],
                                               hp["mu_min"], hp["mu_max"])
    print("Predicting the {n_s_hifi} corresponding solutions")
    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)
    U_pred_hifi_mean = model.restruct(U_pred_hifi_mean[0], no_s=True), model.restruct(U_pred_hifi_mean[1], no_s=True)
    U_pred_hifi_std = model.restruct(U_pred_hifi_std[0], no_s=True), model.restruct(U_pred_hifi_std[1], no_s=True)

    # Plot against test and save
    return plot_results(U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                        train_res, hp, no_plot)


if __name__ == "__main__":
    # Custom hyperparameters as command-line arg
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = yaml.load(HPFile)
    # Default ones
    else:
        from hyperparams import HP

    main(HP, gen_test=False, use_cached_dataset=False)
    # main(HP, gen_test=False, use_cached_dataset=True)
