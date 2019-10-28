import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import pickle
from pyDOE import lhs
import time

eqnPath = "1d-burgers2"
sys.path.append(eqnPath)
from hyperparams import hp
from dataprep import prep_data, burgers_u
from regression import RegNN

sys.path.append("utils")
from handling import pack_layers


def predict_and_assess(model, X_v_val, U_val, V, hp, comp_time=False):
    v_pred = model.predict(X_v_val)
    print(f"Error calculated on nn_s_train = {X_v_val.shape[0]} samples" +
          f" ({int(100 * hp['train_val_ratio'])}%)")

    # Retrieving the function with the predicted coefficients
    U_pred = V.dot(v_pred.T)

    # Restruct
    n_s_val = int((1. - hp["train_val_ratio"]) * hp["n_s"])
    U_pred_struct = restruct(U_val, hp["n_x"], hp["n_t"], n_s_val)
    U_val_struct = restruct(U_pred, hp["n_x"], hp["n_t"], n_s_val)

    if comp_time:
        # Timing the new ROM
        perform_time_comp(model, V, hp)

    return U_val_struct, U_pred_struct


def perform_time_comp(model, V, hp):
    mu = (hp["mu_min"] + hp["mu_max"]) / 2
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
    tT = t.reshape((hp["n_t"], 1))
    X = np.hstack((tT, np.ones_like(tT)*mu))

    print("Getting analytical solution")
    start_ana = time.time()
    U_ana = burgers_u(x, t, mu)
    print(time.time() - start_ana)

    print("Getting reduced order solution")
    start_rom = time.time()
    U_rom = V.dot(model.predict(X).T)
    print(time.time() - start_rom)

    # # Plotting one prediction
    # plt.plot(x, U_ana[:, 50])
    # plt.plot(x, U_rom[:, 50])
    # plt.show()


def restruct(U, n_x, n_t, n_s): 
    U_struct = np.zeros((n_x, n_t, n_s))
    for i in range(n_s):
        s = n_t * i
        e = n_t * (i + 1)
        U_struct[:, :, i] = U[:, s:e]
    return U_struct


if __name__ == "__main__":
    X_v_train, v_train, X_v_val, v_val, \
        lb, ub, V, U_val = prep_data(hp, use_cache=True)
        
    hp["layers"] = pack_layers(X_v_train.shape[1], hp["h_layers"],
                               X_v_train.shape[1])
    regnn = RegNN.load_from(os.path.join(eqnPath, "cache", "model.h5"),
                       hp, lb, ub)

    predict_and_assess(regnn, X_v_val, U_val, V, hp, comp_time=True)
