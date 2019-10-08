import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
import os
from tqdm import tqdm
from deap.benchmarks import shekel


def scarcify(X, u, N):
    idx = np.random.choice(X.shape[0], N, replace=False)
    # return X[idx, :], u[idx, :], 
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[idx, :], u[idx, :], X[mask, :], u[mask, :] 


def prep_data(n_e, n_t, bet_count=0, gam_count=3):
    # Shekel parameters (t=10-sized)
    bet = 1/10 * np.array([[1, 2, 2, 4, 4, 6, 3, 7, 5, 5]]).T
    gam = 1. * np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7]]).T
    mu = (bet, gam)
    
    # Perturbations
    bet_var = np.hstack((bet, 0.1 * bet))[:bet_count, :]
    gam_var = np.hstack((gam, 0.1 * gam))[:gam_count, :]
    p_var = np.vstack((bet_var, gam_var))
    
    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling")
    pbar = tqdm(total=100)
    X = lhs(n_t, p_var.shape[0]).T
    pbar.update(50)
    lb = p_var[:, 0] - np.sqrt(3)*p_var[:, 1]
    ub = p_var[:, 0] + np.sqrt(3)*p_var[:, 1]
    X_U_rb = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Creating the snapshots
    print(f"Generating {n_t} corresponding snapshots")
    U_h = np.zeros((n_e, n_t))
    x = np.linspace(0, 10, n_e)
    for i in tqdm(range(n_t)):
        # Altering the beta params with lhs perturbations
        bet_kxsi = X_U_rb[i, :bet_count]
        bet[0:bet_kxsi.shape[0], 0] = bet_kxsi
        # Altering the gamma params with lhs perturbations
        gam_kxsi = X_U_rb[i, bet_count:]
        gam[0:gam_kxsi.shape[0], 0] = gam_kxsi

        # Calling the Shekel function
        U_h[:, i] = -shekel(x[None, :], gam, bet)[0]

    return U_h, X_U_rb, lb, ub

def get_pod_bases(U_h, n_e, n_t, eps=1e-10,
                  do_plots=False, write_file=False, verbose=False):
    start_time = time.time()
    
    # Performing SVD
    W, D, ZT = np.linalg.svd(U_h, full_matrices=False)
    
    # Getting MATLAB-like orientation
    Z = ZT.T
    
    # Storing eigenvalues and their sum
    lambdas = D**2
    sum_lambdas = np.sum(lambdas)
    
    # Finding n_L
    n_L = 0
    sum_lambdas_trunc = 0.
    for i in range(n_t):
        sum_lambdas_trunc += lambdas[i]
        n_L += 1
        if sum_lambdas_trunc/sum_lambdas >= (1 - eps):
            break
    
    # Truncating according to n_L
    lambdas_trunc = lambdas[0:n_L]
    
    # Constructiong the reduced POD base V
    V = np.zeros((n_e, n_L))
    for i in range(n_L):
        V[:, i] = U_h.dot(Z[:, i]) / np.sqrt(lambdas_trunc[i])
    
    if verbose:
        # Number of solutions and Stopping parameter
        print("n_t: ", n_t)
        print("eps: ", eps)
        
        print(f"Elapsed time is {time.time() - start_time} seconds.")
        print(f"L = {n_L}")
    
    if write_file:
        name = f"shek_Pod_bases_lhs_nxy_{n_e}_ns_{n_t}_epsi_{eps}.txt"
        np.savetxt(os.path.join("1d-shekel", "results", name),
                   V, delimiter="   ")
        print(f"Written {name}")
    
    if do_plots: 
        x = np.linspace(0, 10, n_e)
        for i in range(U_h.shape[1]):
            plt.plot(x, U_h[:, i])
        plt.plot(x, np.mean(U_h, axis=1))
        plt.plot(x, np.std(U_h, axis=1))
        plt.show()
        
        for i in range(n_L):
            plt.plot(x, V[:, i])
        plt.show()

    return V

