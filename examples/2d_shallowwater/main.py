""" POD-NN modeling for 2D inviscid Shallow Water Equations."""

import sys
import os
import yaml
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import read_space_sol_input_mesh
from podnn.plotting import genresultdir

from plot import plot_results


def main(resdir, hp, save_cache=False, use_cached_dataset=False):
    """Full example to run POD-NN on 2d_shallowwater."""

    if not use_cached_dataset:
        # Getting data from the files
        mu_path = os.path.join("data", f"INPUT_{hp['n_s']}_Scenarios.txt")
        x_u_mesh_path = os.path.join("data", f"SOL_FV_{hp['n_s']}_Scenarios.txt")
        x_mesh, u_mesh, X_v = \
            read_space_sol_input_mesh(hp["n_s"], hp["mesh_idx"], x_u_mesh_path, mu_path)
        np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
    else:
        x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))
        u_mesh = None
        X_v = None

    # Init the model
    model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

    # Generate the dataset from the mesh and params
    
    X_v_train, v_train, X_v_test, \
        _, U_test = model.convert_dataset(u_mesh, X_v,
                                          hp["train_val"], hp["eps"],
                                          save_cache=save_cache, use_cache=use_cached_dataset)

    # Train
    model.initVNNs(hp["n_M"], hp["h_layers"],
                   hp["lr"], hp["lambda"], hp["adv_eps"], hp["norm"])
    train_res = model.train(X_v_train, v_train, hp["epochs"],
                            hp["train_val"], freq=hp["log_frequency"])


    # Predict and restruct
    U_pred, U_pred_sig = model.predict(X_v_test)
    U_pred_mean = (model.restruct(U_pred.mean(-1), no_s=True),
                   model.restruct(U_pred_sig.mean(-1), no_s=True))
    U_pred_var = ((U_pred_sig - model.pod_sig[:, np.newaxis])**2 + U_pred ** 2).mean(-1) - U_pred.mean(-1) ** 2
    U_pred_std = (model.restruct(U_pred.std(-1), no_s=True),
                       model.restruct(np.sqrt(U_pred_var), no_s=True))
    U_test = model.restruct(U_test)
    sigma_pod = model.pod_sig.mean()

    # Time for one pred
    # import time
    # st = time.time()
    # model.predict(X_v_test[0:1])
    # print(f"{time.time() - st} sec taken for prediction")
    # exit(0)

    # Plot and save the results
    return plot_results(x_mesh, U_test, U_pred_mean, U_pred_std, sigma_pod, resdir, train_res[0], hp)

if __name__ == "__main__":
    # Custom hyperparameters as command-line arg
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = yaml.load(HPFile)
    # Default ones
    else:
        from hyperparams import HP

    resdir = genresultdir()
    main(resdir, HP, save_cache=False, use_cached_dataset=False)
