import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

eqnPath = "1d-burgers2"
sys.path.append(eqnPath)
from hyperparams import hp
from dataprep import prep_data

sys.path.append("utils")
from metrics import error_podnn, error_pod
from neuralnetwork import NeuralNetwork
from logger import Logger
from handling import scarcify, pack_layers


class RegNN(NeuralNetwork):
    def predict_u(self, t, mu, V):
        v_pred = self.predict(np.reshape([t, mu], (1, 2)))
        return V.dot(v_pred.T)


def create_model_and_train(X_v_train, v_train,
                           X_v_val, v_val,
                           hp, ub, lb, V, U_val):
    # Sizes
    n_L = V.shape[1]
    n_d = X_v_train.shape[1]

    # Creating the neural net model, and logger
    # In: (t, mu)
    # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
    hp["layers"] = pack_layers(n_d, hp["h_layers"], n_L)
    logger = Logger(hp)
    regnn = RegNN(hp, logger, ub, lb)

    # Setting the error function
    def error_val():
        v_pred = regnn.predict(X_v_val)
        return error_podnn(U_val, V.dot(v_pred.T))
    logger.set_error_fn(error_val)

    # Training
    regnn.fit(X_v_train, v_train)

    # Saving
    regnn.save_to(os.path.join(eqnPath, "cache", "model.h5"))
    
    return regnn
   

if __name__ == "__main__":
    X_v_train, v_train, X_v_val, v_val, \
        lb, ub, V, U_val = prep_data(hp, use_cache=True)
        
    create_model_and_train(X_v_train, v_train,
                                   X_v_val, v_val,
                                   hp, lb, ub, V, U_val)
