""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

import sys
import json
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import error_podnn, mse
from podnn.mesh import read_space_sol_input_mesh

from plots import plot_results


def main(hp, use_cached_dataset=False,
         use_trained_network=False):
    """Full example to run POD-NN on 2d_shallowwater."""

    if not use_cached_dataset:
        # Getting data from the files
        mu_path = os.path.join("data", "INPUT_100_Scenarios.txt")
        x_u_mesh_path = os.path.join("data", "SOL_FV_100_Scenarios.txt")
        # Each line is: [i, x_i, y_i, z_i(unused), h_i, eta_i, (hu)_i, (hv)_i]
        x_mesh, u_mesh, X_v = \
            read_space_sol_input_mesh(hp["n_s"], hp["mesh_x_idx"], x_u_mesh_path, mu_path)
        np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
    else:
        x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))
        u_mesh = None
        X_v = None
    
    # x_mesh_unique, c = np.unique(x_mesh[:, 1], return_counts=True)
    # print(x_mesh_unique[c > 1])
    # print(x_mesh_unique[c > 1].shape)
    # exit(0)

    # np.savetxt(os.path.join("cache", "test.csv"), x_mesh[:, 1:])
    # x_mesh_tup = [tuple(row[1:]) for row in x_mesh]
    # x_mesh_tup_dupfree = list(dict.fromkeys(x_mesh_tup))
    # print(x_mesh_tup[0], x_mesh_tup_dupfree[0])
    # print(len(x_mesh_tup), len(x_mesh_tup_dupfree))
    # exit(0)

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
            v_val_pred = model.predict_v(X_v_val)
            # return mse(v_val, v_val_pred)
            return error_podnn(v_val, v_val_pred)
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
    return plot_results(x_mesh, U_val, U_pred, hp)

if __name__ == "__main__":
    # Custom hyperparameters as command-line arg
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = json.load(HPFile)
    # Default ones
    else:
        from hyperparams import HP

    # main(HP, use_cached_dataset=False, use_trained_network=False)
    main(HP, use_cached_dataset=True, use_trained_network=False)
    # main(HP, use_cached_dataset=True, use_trained_network=True)
