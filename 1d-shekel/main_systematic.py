import sys
import os
import json
import numpy as np
import tensorflow as tf

np.random.seed(1111)
tf.random.set_seed(1111)

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
sys.path.append("utils")
from pod import get_pod_bases
from metrics import error_podnn
from neuralnetwork import NeuralNetwork
from logger import Logger
from shekelutils import prep_data, plot_results, get_test_data
from handling import restruct, scarcify


# HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Space (dx = 1/30, n_x = 10/dx)
    hp["n_x"] = 300
    # Snapshots count
    hp["n_s"] = [50, 200, 600, 1000, 2000]
    # Train/Val repartition
    hp["train_val_ratio"] = 0.5
    # POD stopping param
    hp["eps"] = 1e-10
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = [1000, 5000, 20000, 50000, 100000]
    hp["tf_lr"] = 0.002
    hp["tf_decay"] = 0.
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    hp["lambda"] = 1e-4
    # Batch size for mini-batch training (0 means full-batch)
    hp["batch_size"] = 0
    # Frequency of the logger
    hp["log_frequency"] = 1000
    # Shekel params
    hp["bet_count"] = 10
    hp["gam_count"] = 10

if __name__ == "__main__":
    # Unpacking hp
    n_x = hp["n_x"]

    # Getting test data
    _, U_test_mean, U_test_std = get_test_data()
    errors_test_mean = np.zeros((len(hp["n_s"]), len(hp["tf_epochs"])))
    errors_test_std = np.zeros((len(hp["n_s"]), len(hp["tf_epochs"])))

    for i_n_s, n_s in enumerate(hp["n_s"]):
        print(f"For n_s={n_s}...")
        for i_tf_epochs, tf_epochs in enumerate(hp["tf_epochs"]):
            print(f"For tf_epochs={tf_epochs}")

            # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
            # u_rb are the reduced coefficients we're looking for
            U_h_star, X_U_rb_star, lb, ub = prep_data(n_x, n_s,
                hp["bet_count"], hp["gam_count"])
            V = get_pod_bases(U_h_star, hp["eps"])

            # Sizes
            n_L = V.shape[1]
            n_d = X_U_rb_star.shape[1]

            # Projecting
            U_rb_star = (V.T.dot(U_h_star)).T

            # Splitting data
            n_s_train = int(hp["train_val_ratio"] * n_s)
            n_s_val = n_s - n_s_train
            X_U_rb_train, U_rb_train, X_U_rb_val, U_rb_val = \
                    scarcify(X_U_rb_star, U_rb_star, n_s_train)
            U_h_val = V.dot(U_rb_val.T)

            # Creating the neural net model, and logger
            # In: (gam_0, bet_1, ..., bet_m, gam_0, bet_1, ..., bet_n)
            # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
            hp["layers"] = [n_d, 40, 60, n_L]
            logger = Logger(hp)
            model = NeuralNetwork(hp, logger, ub, lb)

            # Setting the error function on validation data (E_va)
            def error_val():
                U_rb_pred = model.predict(X_U_rb_val)
                return error_podnn(U_h_val, V.dot(U_rb_pred.T))
            logger.set_error_fn(error_val)

            # Training
            model.fit(X_U_rb_train, U_rb_train, tf_epochs)

            # Predicting the coefficients
            U_rb_pred = model.predict(X_U_rb_val)

            # Retrieving the function with the predicted coefficients
            U_h_pred = V.dot(U_rb_pred.T)

            # Saving the results
            U_pred_mean = np.mean(U_h_pred, axis=1)
            U_pred_std = np.std(U_h_pred, axis=1)
            error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
            error_test_std = 100 * error_podnn(U_test_std, U_pred_std)

            errors_test_mean[i_n_s, i_tf_epochs] = error_test_mean
            errors_test_std[i_n_s, i_tf_epochs] = error_test_std

    np.savetxt(os.path.join(eqnPath, "results", "systematic", "n_s.csv"),
               hp["n_s"])
    np.savetxt(os.path.join(eqnPath, "results", "systematic", "tf_epochs.csv"),
               hp["tf_epochs"])
    np.savetxt(os.path.join(eqnPath, "results", "systematic", "err_t_mean.csv"),
               errors_test_mean)
    np.savetxt(os.path.join(eqnPath, "results", "systematic", "err_t_std.csv"),
               errors_test_std)
