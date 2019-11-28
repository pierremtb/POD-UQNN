""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

import sys
import os
import yaml
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import error_podnn_rel
from podnn.mesh import read_space_sol_input_mesh

from plots import plot_results


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

    # Create the POD-NN model
    model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])

    # Generate the dataset from the mesh and params
    X_v_train, v_train, \
        X_v_test, _, \
        U_test = model.convert_dataset(u_mesh, X_v,
                                      hp["train_val_test"], hp["eps"],
                                      use_cache=use_cached_dataset)

    U_test_mean = np.mean(U_test, axis=-1)
    U_test_std = np.nanstd(U_test, axis=-1)

    # Create the model and train
    train_res = model.train(X_v_train, v_train, hp["h_layers"],
                            hp["epochs"], hp["lr"], hp["lambda"],
                            hp["train_val_test"], 
                            hp["decay"], hp["log_frequency"])

    # Predict and restruct
    U_pred = model.predict(X_v_test)
    U_pred = model.restruct(U_pred)
    U_test = model.restruct(U_test)

    error_test_mean, error_test_std = error_podnn_rel(U_test, U_pred)
    print("--")
    print(f"Error on the mean test HiFi LHS solution: {error_test_mean:4f}")
    print(f"Error on the stdd test HiFi LHS solution: {error_test_std:4f}")
    print("--")

    # Time for one pred
    # import time
    # st = time.time()
    # model.predict(X_v_test[0:1])
    # print(f"{time.time() - st} sec taken for prediction")
    # exit(0)

    # Plot and save the results
    return plot_results(x_mesh, U_test, U_pred, hp, train_res)

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
