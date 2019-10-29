import numpy as np
import sys
import os
from tqdm import tqdm
import pickle
from pyDOE import lhs
from deap.benchmarks import shekel

eqnPath = "1d-shekel"
sys.path.append(eqnPath)

sys.path.append("utils")
from pod import get_pod_bases


def shekel_u(x, gam, bet):
    return -shekel(x[None, :], gam, bet)[0]


def prep_data(hp, use_cache=False, save_cache=False):
    cache_path = os.path.join(eqnPath, "cache", "prep_data.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print("Loaded cached data")
            return pickle.load(f)

    # Shekel parameters (t=10-sized)
    bet = 1/10 * np.array([[1, 2, 2, 4, 4, 6, 3, 7, 5, 5]]).T
    gam = 1. * np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7]]).T
    
    # Perturbations
    bet_var = np.hstack((bet, 0.1 * bet))[:hp["bet_count"], :]
    gam_var = np.hstack((gam, 0.1 * gam))[:hp["gam_count"], :]
    p_var = np.vstack((bet_var, gam_var))
    
    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling")
    pbar = tqdm(total=100)
    X = lhs(hp["n_s"], p_var.shape[0]).T
    pbar.update(50)
    lb = p_var[:, 0] - np.sqrt(3)*p_var[:, 1]
    ub = p_var[:, 0] + np.sqrt(3)*p_var[:, 1]
    X_v = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Creating the snapshots
    print(f"Generating {hp['n_s']} corresponding snapshots")
    U = np.zeros((hp["n_x"], hp["n_s"]))
    x = np.linspace(0, 10, hp["n_x"])
    for i in tqdm(range(hp["n_s"])):
        # Altering the beta params with lhs perturbations
        bet_kxsi = X_v[i, :hp["bet_count"]]
        bet[0:bet_kxsi.shape[0], 0] = bet_kxsi
        # Altering the gamma params with lhs perturbations
        gam_kxsi = X_v[i, hp["bet_count"]:]
        gam[0:gam_kxsi.shape[0], 0] = gam_kxsi

        # Calling the Shekel function
        U[:, i] = shekel_u(x, gam, bet)

    # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
    # u_rb are the reduced coefficients we're looking for
    V = get_pod_bases(U, hp["eps"])

    # Projecting
    v = (V.T.dot(U)).T

    # Splitting the dataset (X_v, v)
    nn_s_train = int(hp["train_val_ratio"] * hp["n_s"])
    X_v_train, v_train = X_v[:nn_s_train, :], v[:nn_s_train, :]
    X_v_val, v_val = X_v[nn_s_train:, :], v[nn_s_train:, :]

    # Creating the validation snapshots matrix
    U_val = V.dot(v_val.T)

    if save_cache:
        with open(cache_path, "wb") as f:
            pickle.dump((X_v_train, v_train, X_v_val, v_val,
                         lb, ub, V, U_val), f)

    return U, X_v, lb, ub, V, U_val

if __name__ == "__main__":
    prep_data(hp, fast_pod=True, save_cache=True)

