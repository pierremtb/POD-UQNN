import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
import os

# Defining the u_h function (here Shekel)
def u_h(x, mu):
    bet, gam = mu
    S_i = np.zeros_like(x)
    for j, x_j in enumerate(x):
        S_i[j] = 0.
        for p in range(bet.shape[1]):
            S_i[j] -= 1/((x_j-gam[0, p])**2 + bet[0, p])
    return S_i
    
def get_pod_bases(eps=1e-10, n_t=100, do_plots=False, write_file=False, verbose=False):
    start_time = time.time()
    
    # Space params
    dx = 1/30
    n_e = int(10/dx)
    
    # Shekel parameters (t=10-sized)
    bet = 1/10 * np.array([[1, 2, 2, 4, 4, 6, 3, 7, 5, 5]]).T
    gam = 1. * np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7]]).T
    mu = (bet, gam)
    
    # Perturbations
    p_var = np.array([
        [4, 0.4],
        [1, 0.1],
        [8, 0.8],
    ])
    
    # LHS sampling (first uniform, then perturbated)
    X = lhs(n_t, p_var.shape[0]).T
    lb = p_var[:, 0] - np.sqrt(3)*p_var[:, 1]
    ub = p_var[:, 0] + np.sqrt(3)*p_var[:, 1]
    R_var = lb + (ub - lb)*X
    
    # Creating the snapshots
    S = np.zeros((n_e, n_t))
    x = np.linspace(0, 10, n_e)
    for i in range(n_t):
        # Altering the parameters with lhs perturbations
        kxsi = R_var[i, :]
        mu[1][0:kxsi.shape[0], 0] = kxsi
        # Calling the Shekel function
        S[:, i] = u_h(x, mu)
    
    # Performing SVD
    W, D, ZT = np.linalg.svd(S, full_matrices=False)
    
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
        V[:, i] = S.dot(Z[:, i]) / np.sqrt(lambdas_trunc[i])
    
    if verbose:
        # Number of solutions and Stopping parameter
        print("n_t: ", n_t)
        print("eps: ", eps)
        
        print(f"Elapsed time is {time.time() - start_time} seconds.")
        print(f"L = {n_L}")
    
    if write_file:
        name = f"shek_Pod_bases_lhs_nxy_{n_e}_ns_{n_t}_epsi_{eps}.txt"
        np.savetxt(os.path.join("1d-shekel", "results", name), V, delimiter="   ")
        print(f"Written {name}")
    
    if do_plots:    
        for i in range(S.shape[1]):
            plt.plot(x, S[:, i])
        plt.plot(x, np.mean(S, axis=1))
        plt.plot(x, np.std(S, axis=1))
        plt.show()
        
        for i in range(n_L):
            plt.plot(x, V[:, i])
        plt.show()

    return x, V


if __name__ == "__main__":
    get_pod_bases(eps=1e-4, n_t=1000, do_plots=True, write_file=True, verbose=True)
