""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

import sys
import os
import yaml
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_mean_std, re
from podnn.mesh import read_space_sol_input_mesh

from plot import plot_results


def main(hp, use_cached_dataset=False):
    """Full example to run POD-NN on 2d_shallowwater."""

    if not use_cached_dataset:
        # Getting data from the files
        mu_path = os.path.join("data", f"INPUT_{hp['n_s']}_Scenarios.txt")
        x_u_mesh_path = os.path.join("data", f"SOL_FV_{hp['n_s']}_Scenarios.txt")
        x_mesh, u_mesh, X_v = \
            read_space_sol_input_mesh(hp["n_s"], hp["mesh_idx"], x_u_mesh_path, mu_path)
        np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
    else:
        x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
        u_mesh = None
        X_v = None

    # Init the model
    model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])

    # Generate the dataset from the mesh and params
    
    X_v_train, v_train, \
        X_v_test, _, \
        U_test = model.convert_dataset(u_mesh, X_v,
                                       hp["train_val_test"], hp["eps"],
                                       use_cache=use_cached_dataset)

    # Train
    model.initNN(hp["h_layers"], hp["lr"], hp["lambda"])
    train_res = model.train(X_v_train, v_train, hp["epochs"],
                            hp["train_val_test"], freq=hp["log_frequency"])

    # Predict and restruct
    U_pred = model.predict(X_v_test)
    U_pred = model.restruct(U_pred)
    U_test = model.restruct(U_test)

    # Compute relative error
    error_test_mean, error_test_std = re_mean_std(U_test, U_pred)
    print(f"Test relative error: mean {error_test_mean:4f}, std {error_test_std:4f}")

    mu_path = os.path.join("data", f"INPUT_{hp['n_s_hifi']}_Scenarios.txt")
    x_u_mesh_path = os.path.join("data", f"SOL_FV_{hp['n_s_hifi']}_Scenarios.txt")
    _, u_mesh_test_hifi, X_v_test_hifi = \
        read_space_sol_input_mesh(hp["n_s"], hp["mesh_idx"], x_u_mesh_path, mu_path)
    U_test_hifi = model.u_mesh_to_U(u_mesh_test_hifi, hp["n_s_hifi"])
    U_test_hifi_mean, U_test_hifi_std = U_test_hifi.mean(-1), np.nanstd(U_test_hifi, -1)

    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)
    error_test_hifi_mean = re(U_pred_hifi_mean, U_test_hifi_mean)
    error_test_hifi_std = re(U_pred_hifi_std, U_test_hifi_std)
    print(f"Hifi Test relative error: mean {error_test_hifi_mean:4f}, std {error_test_hifi_std:4f}")

    # Restruct for plotting
    U_test_hifi_mean = model.restruct(U_test_hifi_mean, no_s=True)
    U_test_hifi_std = model.restruct(U_test_hifi_std, no_s=True)
    U_pred_hifi_mean = model.restruct(U_pred_hifi_mean, no_s=True)
    U_pred_hifi_std = model.restruct(U_pred_hifi_std, no_s=True)

    # Time for one pred
    # import time
    # st = time.time()
    # model.predict(X_v_test[0:1])
    # print(f"{time.time() - st} sec taken for prediction")
    # exit(0)

    # Plot and save the results
    return plot_results(x_mesh, U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                        U_test_hifi_mean, U_test_hifi_std,
                        train_res, HP=hp, export_vtk=True, export_txt=False)

if __name__ == "__main__":
    # Custom hyperparameters as command-line arg
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = yaml.load(HPFile)
    # Default ones
    else:
        from hyperparams import HP

    # main(HP, use_cached_dataset=False)
    main(HP, use_cached_dataset=True)
