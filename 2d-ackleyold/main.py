import sys
import json
import numpy as np
import tensorflow as tf

np.random.seed(1111)
tf.random.set_seed(1111)

eqnPath = "2d-ackley"
sys.path.append(eqnPath)
sys.path.append("utils")
from pod import get_pod_bases
from metrics import error_podnn, error_pod
from neuralnetwork import NeuralNetwork
from logger import Logger
from ackleyutils import prep_data, plot_results
from handling import restruct, scarcify, pack_layers


# HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Space
    hp["n_x"] = 400
    hp["n_y"] = 400
    hp["x_min"] = -5.
    hp["x_max"] = 5.
    hp["y_min"] = -5.
    hp["y_max"] = 5.
    # Snapshots count
    hp["n_s"] = 1000
    # Train/val split
    hp["train_val_ratio"] = 0.5
    # Deep NN hidden layers topology
    hp["h_layers"] = [64, 64]
    # Batch size for mini-batch training (0 means full-batch)
    hp["batch_size"] = 0
    # POD stopping param
    hp["eps"] = 1e-10
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    # hp["tf_epochs"] = 0
    hp["tf_epochs"] = 100000
    hp["tf_lr"] = 0.002
    hp["tf_decay"] = 0.
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    hp["lambda"] = 1e-4
    # Frequency of the logger
    hp["log_frequency"] = 1000

if __name__ == "__main__":
    n_x = hp["n_x"]
    n_y = hp["n_y"]
    n_s = hp["n_s"]
    x_min = hp["x_min"]
    x_max = hp["x_max"]
    y_min = hp["y_min"]
    y_max = hp["y_max"]

    # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
    # u_rb are the reduced coefficients we're looking for
    X, Y, U_star, X_v_star, lb, ub = prep_data(n_x, n_y, n_s,
                                                    x_min, x_max,
                                                    y_min, y_max)
    V = get_pod_bases(U_star, hp["eps"])

    # Sizes
    n_L = V.shape[1]
    n_d = X_v_star.shape[1]

    # Projecting
    v_star = (V.T.dot(U_star)).T

    # Splitting data
    n_s_train = int(hp["train_val_ratio"] * hp["n_s"])
    n_s_val = n_s - n_s_train
    X_v_train, v_train, X_v_val, v_val = \
            scarcify(X_v_star, v_star, n_s_train)
    U_val = V.dot(v_val.T)

    # Creating the neural net model, and logger
    # In: (mu_0, mu_1, mu_2)
    # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
    hp["layers"] = pack_layers(n_d, hp["h_layers"], n_L)
    logger = Logger(hp)
    model = NeuralNetwork(hp, logger, ub, lb)

    # Setting the error function on validation data (E_va)
    def error_val():
        v_pred = model.predict(X_v_val)
        return error_podnn(U_val, V.dot(v_pred.T))
    logger.set_error_fn(error_val)

    # Training
    model.fit(X_v_train, v_train)

    # Predicting the coefficients
    v_pred = model.predict(X_v_val)
    print(f"Error calculated on n_s_val = {n_s_val} samples" +
          f" ({int(100 * hp['train_val_ratio'])}%)")

    # Retrieving the function with the predicted coefficients
    U_pred = V.dot(v_pred.T)

    # Restructuring
    U_val_struct = np.reshape(U_val, (n_x, n_y, n_s_val))
    U_pred_struct = np.reshape(U_pred, (n_x, n_y, n_s_val))

    # Plotting and saving the results
    plot_results(U_val_struct, U_pred_struct, hp, eqnPath)
    plot_results(U_val_struct, U_pred_struct, hp)
