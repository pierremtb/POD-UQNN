import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
sys.path.append("utils")
from metrics import error_podnn, error_pod
from neuralnetwork import NeuralNetwork
from logger import Logger
from handling import scarcify, pack_layers
from hyperparams import hp
from dataprep import prep_data


def create_model_and_train(X_v_train, v_train,
                           X_v_val, v_val,
                           ub, lb, V, U_val):
    # Sizes
    n_L = V.shape[1]
    n_d = X_v_train.shape[1]

    class PiNeuralNetwork(NeuralNetwork):
        def predict_u(self, t, mu, V):
            v_pred = self.predict(np.reshape([t, mu], (1, 2)))
            return V.dot(v_pred.T)

    # Creating the neural net model, and logger
    # In: (t, mu)
    # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
    hp["layers"] = pack_layers(n_d, hp["h_layers"], n_L)
    logger = Logger(hp)
    model = PiNeuralNetwork(hp, logger, ub, lb)

    # Setting the error function
    def error_val():
        v_pred = model.predict(X_v_val)
        return error_podnn(U_val, V.dot(v_pred.T))
    logger.set_error_fn(error_val)

    # Training
    model.fit(X_v_train, v_train)

    # Saving
    model.model.save(os.path.join(eqnPath, "cache", "model.h5"))
    
    return model
   

if __name__ == "__main__":
    np.random.seed(1111)
    tf.random.set_seed(1111)

    X_v_train, v_train, X_v_val, v_val, \
        lb, ub, V, U_val = prep_data(
            hp["n_x"], hp["x_min"], hp["x_max"],
            hp["n_t"], hp["t_min"], hp["t_max"],
            hp["n_s"], hp["mu_mean"],
            hp["train_val_ratio"], hp["eps"],
            use_cache=True, save_cache=True)
        
    # NN-REGRESSION TRAINING
    model = create_model_and_train(X_v_train, v_train,
                                   X_v_val, v_val,
                                   lb, ub, V, U_val)
