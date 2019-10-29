import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
from dataprep import prep_data
from shekelutils import plot_results, prep_data

sys.path.append("utils")
from pod import get_pod_bases
from metrics import error_podnn, error_pod
from neuralnetwork import NeuralNetwork
from logger import Logger
from handling import scarcify, pack_layers


# HYPER PARAMETERS
if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    from hyperparams import hp

# DATA PREPARATION WITH POD
U_star, X_v_star, lb, ub, V, U_val = prep_data(hp)

# NN-REGRESSION TRAINING
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
