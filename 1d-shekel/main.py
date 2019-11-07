import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EQN_PATH = "1d-shekel"
sys.path.append(EQN_PATH)
from datagen import u
from plots import plot_results

sys.path.append("utils")
from podnn import PodnnModel
from metrics import error_podnn
from mesh import create_linear_mesh


def main(HP, no_plot=False):
    # Create linear space mesh
    x_mesh = create_linear_mesh(HP["x_min"], HP["x_max"], HP["n_x"])

    # Extend the class and init the model
    class Burgers2PodnnModel(PodnnModel):
        def u(self, X, t, mu):
            return u(X, t, mu)
    model = Burgers2PodnnModel(HP["n_v"], x_mesh, HP["n_t"], EQN_PATH)

    # Generate the dataset from the mesh and params
    X_v_train, v_train, \
        X_v_val, v_val, \
        U_val = model.generate_dataset(HP["mu_min"], HP["mu_max"],
                                    HP["n_s"],
                                    HP["train_val_ratio"],
                                    HP["eps"])

    # Train
    def error_val():
        U_pred = model.predict(X_v_val)
        return error_podnn(U_val, U_pred)
    model.train(X_v_train, v_train, error_val, HP["h_layers"],
                HP["epochs"], HP["lr"], HP["lambda"]) 

    # Predict and restruct
    U_pred = model.predict(X_v_val)

    # Plot against test and save
    return plot_results(U_val, U_pred, HP, EQN_PATH, no_plot)


if __name__ == "__main__":
    # HYPER PARAMETERS
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = json.load(HPFile)
    else:
        from hyperparams import HP
    main(HP)