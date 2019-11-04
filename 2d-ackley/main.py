import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

eqnPath = "2d-ackley"
sys.path.append(eqnPath)
from datagen import u
from plots import plot_results

sys.path.append("utils")
from podnn import PodnnModel
from metrics import error_podnn
from mesh import create_linear_mesh


def main(hp, no_plot=False):
    # Create linear space mesh
    x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"],
                                hp["y_min"], hp["y_max"], hp["n_y"])

    # Extend the class and init the model
    class AckleyPodnnModel(PodnnModel):
        def u(self, X, t, mu):
            return u(X, t, mu)
    model = AckleyPodnnModel(hp["n_v"], x_mesh, hp["n_t"], eqnPath)

    # Generate the dataset from the mesh and params
    X_v_train, v_train, \
        X_v_val, v_val, \
        U_val = model.generate_dataset(hp["mu_min"], hp["mu_max"],
                                    hp["n_s"],
                                    hp["train_val_ratio"],
                                    hp["eps"],
                                    use_cache=True,
                                    save_cache=True)

    # Train
    def error_val():
        U_pred = model.predict(X_v_val)
        return error_podnn(U_val, U_pred)
    model.train(X_v_train, v_train, error_val, hp["h_layers"],
                hp["epochs"], hp["lr"], hp["lambda"]) 

    # Predict and restruct
    U_pred = model.predict(X_v_val)
    
    # Plot against test and save
    return plot_results(U_val, U_pred, hp, eqnPath, no_plot)


if __name__ == "__main__":
    # HYPER PARAMETERS
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as hpFile:
            hp = json.load(hpFile)
    else:
        from hyperparams import hp
    main(hp)
