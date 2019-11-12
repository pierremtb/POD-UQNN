""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

import os
import sys
import json
import numpy as np

from podnn.podnnmodel import PodnnModel
from podnn.metrics import error_podnn
from podnn.mesh import read_space_sol_input_mesh
from plots import plot_results


def main(hp, use_cached_dataset=False,
         use_trained_network=False, no_plot=False):
    """Full example to run POD-NN on 2d_shallowwater."""

    use_cached_dataset = True
    use_trained_network = True
    # USE_TRAINED_NETWORK = False

    if not use_cached_dataset:
        # Getting data from the files
        mu_path = os.path.join("data", "INPUT_100_Scenarios.txt")
        x_u_mesh_path = os.path.join("data", "SOL_FV_100_Scenarios.txt")
        # Each line is: [i, x_i, y_i, z_i(unused), h_i, eta_i, (hu)_i, (hv)_i]
        idx_x = [1, 2]
        idx_u = [4, 6, 7] 
        hp["n_v"] = len(idx_u)
        x_mesh, u_mesh, X_v = \
            read_space_sol_input_mesh(hp["n_s"], idx_x, idx_u, x_u_mesh_path, mu_path)
        np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
    else:
        x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
        u_mesh = None
        X_v = None

    # Create the POD-NN model
    model = PodnnModel(hp["n_v"], x_mesh, hp["n_t"])

    # Generate the dataset from the mesh and params
    X_v_train, v_train, \
        X_v_val, v_val, \
        U_val = model.convert_dataset(u_mesh, X_v,
                                    hp["train_val_ratio"], hp["eps"],
                                    use_cache=use_cached_dataset, save_cache=True)

    # Create the model and train
    if not use_trained_network:
        def error_val():
            """Defines the error metric for in-training validation."""
            U_val_pred = model.predict(X_v_val)
            return error_podnn(U_val, U_val_pred)
        model.train(X_v_train, v_train, error_val, hp["h_layers"],
                    hp["epochs"], hp["lr"], hp["lambda"])
    else:
        model.load_trained_cache()

    # Predict and restruct
    U_pred = model.predict(X_v_val)
    U_pred = model.restruct(U_pred)
    U_val = model.restruct(U_val)

    # Time for one pred
    # import time
    # st = time.time()
    # model.predict(X_v_val[0:1])
    # print(f"{time.time() - st} sec taken for prediction")
    # exit(0)

    # Plot and save the results
    return plot_results(x_mesh, U_val, U_pred, hp, no_plot)

if __name__ == "__main__":
    # Custom hyperparameters as command-line arg
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = json.load(HPFile)
    # Default ones
    else:
        from hyperparams import HP

    main(HP, use_cached_dataset=False, use_trained_network=True)
