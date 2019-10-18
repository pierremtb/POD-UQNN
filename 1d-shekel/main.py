import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1111)
tf.random.set_seed(1111)

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
sys.path.append("utils")
from pod import get_pod_bases
from metrics import error_podnn, error_pod
from neuralnetwork import NeuralNetwork
from logger import Logger
from shekelutils import plot_results, prep_data
from handling import scarcify, pack_layers


# HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Space (dx = 1/30, n_x = 10/dx)
    hp["n_x"] = 300
    # Snapshots count
    hp["n_s"] = 1000
    # POD stopping param
    hp["eps"] = 1e-10
    # Train/val split
    hp["train_val_ratio"] = 0.7
    # Deep NN hidden layers topology
    hp["h_layers"] = [64, 64]
    # Batch size for mini-batch training (0 means full-batch)
    hp["batch_size"] = 0
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 70000
    hp["tf_lr"] = 0.003
    hp["tf_decay"] = 0.
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    hp["lambda"] = 1e-6
    # Frequency of the logger
    hp["log_frequency"] = 1000
    # Shekel params
    hp["bet_count"] = 10
    hp["gam_count"] = 10

if __name__ == "__main__":
    # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
    # u_rb are the reduced coefficients we're looking for
    U_star, X_v_star, lb, ub = prep_data(hp["n_x"], hp["n_s"], hp["bet_count"], hp["gam_count"])
    V = get_pod_bases(U_star, hp["eps"])

    print(f"POD relative error: {100 * error_pod(U_star, V):.4f}%")
    
    # Sizes
    n_L = V.shape[1]
    n_d = X_v_star.shape[1]

    # Projecting
    v_star = (V.T.dot(U_star)).T

    # Splitting data
    n_s_train = int(hp["train_val_ratio"] * hp["n_s"])
    X_v_train, v_train, X_v_val, v_val = \
            scarcify(X_v_star, v_star, n_s_train)
    U_val = V.dot(v_val.T)

    # Creating the neural net model, and logger
    # In: (gam_0, bet_1, ..., bet_m, gam_0, bet_1, ..., bet_n)
    # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
    hp["layers"] = pack_layers(n_d, hp["h_layers"], n_L)
    logger = Logger(hp)
    model = NeuralNetwork(hp, logger, ub, lb)

    # Setting the error function
    def error_val():
        v_pred = model.predict(X_v_val)
        return error_podnn(U_val, V.dot(v_pred.T))
    logger.set_error_fn(error_val)

    # Training
    model.fit(X_v_train, v_train)

    # Predicting the coefficients
    v_pred = model.predict(X_v_val)
    print(f"Error calculated on n_s_train = {n_s_train} samples" +
          f" ({int(100 * hp['train_val_ratio'])}%)")

    # Retrieving the function with the predicted coefficients
    U_pred = V.dot(v_pred.T)

    # Plotting and saving the results
    plot_results(U_val, U_pred, hp, eqnPath)
    plot_results(U_val, U_pred, hp)
