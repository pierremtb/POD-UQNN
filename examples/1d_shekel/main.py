"""POD-NN modeling for 1D Shekel Equation."""

import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.plotting import genresultdir

from genhifi import u, generate_test_dataset
from plot import plot_results


def main(resdir, hp, gen_test=False, use_cached_dataset=False,
         no_plot=False):
    """Full example to run POD-NN on 1d_shekel."""

    if gen_test:
        generate_test_dataset()

    if not use_cached_dataset:
        # Create linear space mesh
        x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
        np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
    else:
        x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))

    # Init the model
    model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

    # Generate the dataset from the mesh and params
    X_v_train, v_train, _, \
        X_v_test, _, U_test = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                                     hp["n_s"],
                                                     hp["train_val_test"],
                                                     eps=hp["eps"], n_L=hp["n_L"],
                                                     u_noise=hp["u_noise"],
                                                     x_noise=hp["x_noise"],
                                                     use_cache=use_cached_dataset)

    # Train
    model.initVNNs(hp["n_M"], hp["h_layers"], 
                   hp["lr"], hp["lambda"], hp["adv_eps"], hp["norm"])
    train_res = model.train(X_v_train, v_train, hp["epochs"],
                            hp["train_val_test"], freq=hp["log_frequency"])
    # Predict and restruct
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
    sigma_pod = model.pod_sig.mean()

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling on the non-spatial params...")
    mu_min_out, mu_min = np.array(hp["mu_min_out"]), np.array(hp["mu_min"])
    # mu_lhs = model.sample_mu(hp["n_s"], mu_min_out, mu_min, linear=True)
    mu_lhs = np.linspace(mu_min_out, mu_min, hp["n_s"])
    n_d = mu_lhs.shape[1]
    n_h = hp["n_v"] * x_mesh.shape[0]
    X_v_test_out, U_test_out, U_test_out_struct, _ = \
        model.create_snapshots(n_d, n_h, u, mu_lhs)
    # Projecting
    U_pred_out, U_pred_out_sig = model.predict(X_v_test_out)

    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    lower = U_pred_out - 2 * U_pred_out_sig
    upper = U_pred_out + 2 * U_pred_out_sig

    for i in [0, 249, 499]:
        plt.fill_between(x, lower[:, i], upper[:, i], 
                            facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
        plt.plot(x, U_pred_out[:, i], "b-")
        plt.plot(x, U_test_out[:, i], "r--")
        plt.show()

    # Plot against test and save
    return plot_results(U_test, U_pred, U_pred_hifi_mean, U_pred_hifi_std, sigma_pod,
                        resdir, train_res[0], hp, no_plot)


if __name__ == "__main__":
    # Custom hyperparameters as command-line arg
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = yaml.load(HPFile)
    # Default ones
    else:
        from hyperparams import HP

    resdir = genresultdir()
    main(resdir, HP, gen_test=False, use_cached_dataset=False)
