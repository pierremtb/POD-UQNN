import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
from pyDOE import lhs
import time

eqnPath = "1d-burgers2"
sys.path.append(eqnPath)
from hyperparams import hp

sys.path.append("utils")
from pod import get_pod_bases


def u_1(x, mu):
    return x / (1 + np.exp(1/(4*mu)*(x**2 - 1/4)))


def u(x, t, mu):
    t0 = np.exp(1 / (8*mu))
    return (x/t) / (1 + np.sqrt(t/t0)*np.exp(x**2/(4*mu*t)))
    

def burgers_u(x, t, mu):
    x, t = np.array(x), np.array(t)
    U = np.zeros((x.shape[0], t.shape[0]))
    U[:, 0] = u_1(x, mu)
    XT, TT = np.meshgrid(x, t[0:])
    X, T = XT.T, TT.T
    U[:, 0:] = u(X, T, mu)
    return U


def prep_data(hp, fast_pod=False, save_cache=False, use_cache=False):
    cache_path = os.path.join(eqnPath, "cache", "prep_data.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print("Loaded cached data")
            return pickle.load(f)

    # Total number of snapshots
    nn_s = hp["n_t"]*hp["n_s"]

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling...")
    pbar = tqdm(total=100)
    X = lhs(hp["n_s"], 1).T
    pbar.update(50)
    lb = hp["mu_min"]
    ub = hp["mu_max"]
    mu_lhs = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()

    # Number of inputs in time plus number of parameters
    n_d = 1 + 1

    # Creating the snapshots
    print(f"Generating {nn_s} corresponding snapshots")
    X_v = np.zeros((nn_s, n_d))
    U = np.zeros((hp["n_x"], nn_s))
    U_struct = np.zeros((hp["n_x"], hp["n_t"], hp["n_s"]))
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
    tT = t.reshape((hp["n_t"], 1))
    for i in tqdm(range(hp["n_s"])):
        # Calling the analytical solution function
        s = hp["n_t"] * i
        e = hp["n_t"] * (i + 1)
        X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))
        U[:, s:e] = burgers_u(x, t, mu_lhs[i, :])
        U_struct[:, :, i] = U[:, s:e]

    # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
    # u_rb are the reduced coefficients we're looking for
    if fast_pod:
        V = get_pod_bases(U_struct, hp["eps"], eps_init_step=hp["eps_init"])
    else:
        V = get_pod_bases(U, hp["eps"])

    # Projecting
    v = (V.T.dot(U)).T
   
    # Splitting the dataset (X_v, v)
    nn_s_train = int(hp["train_val_ratio"] * nn_s)
    X_v_train, v_train = X_v[:nn_s_train, :], v[:nn_s_train, :]
    X_v_val, v_val = X_v[nn_s_train:, :], v[nn_s_train:, :]
   
    # Creating the validation snapshots matrix
    U_val = V.dot(v_val.T)

    if save_cache:
        with open(cache_path, "wb") as f:
            pickle.dump((X_v_train, v_train, X_v_val, v_val,
                         lb, ub, V, U_val), f)

    return X_v_train, v_train, X_v_val, v_val, \
        lb, ub, V, U_val

if __name__ == "__main__":
    prep_data(hp, fast_pod=True, save_cache=True)

