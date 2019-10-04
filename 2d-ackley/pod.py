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

def u_h(x, y, mu):
    return - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
           - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
           + 20 + np.exp(1)


def prep_data(n_x, n_y, n_s):
    # Auckley params mean
    mu = np.array([0., 0., 0.])

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling")
    pbar = tqdm(total=100)
    X = lhs(n_s, mu.shape[0]).T
    pbar.update(50)
    lb = -1. * np.ones_like(mu)
    ub = +1. * np.ones_like(mu)
    X_U_rb = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()
    
    # Creating the snapshots
    print(f"Generating {n_s} corresponding snapshots")
    U_h = np.zeros((n_x, n_y, n_s))
    x = np.linspace(-5, 5, n_x)
    y = np.linspace(-5, 5, n_y)
    X, Y = np.meshgrid(x, y)
    for i in tqdm(range(n_s)):
        mu_i = X_U_rb[i, :]

        # Calling the function
        U_h[:, :, i] = u_h(X, Y, mu_i)

    return U_h, X_U_rb, lb, ub

def get_pod_bases(U_h, n_x, n_y, n_s, eps=1e-10,
                  do_plots=False, write_file=False, verbose=False):
    start_time = time.time()
    
    # Performing SVD
    W, D, ZT = np.linalg.svd(U_h, full_matrices=False)
    print(D)
    print(W.shape, D.shape, ZT.shape)
    
    # Getting MATLAB-like orientation
    Z = ZT.T
    
    # Storing eigenvalues and their sum
    lambdas = D**2
    sum_lambdas = np.sum(lambdas)
    
    # Finding n_L
    n_L = 0
    sum_lambdas_trunc = 0.
    for i in range(n_s):
        sum_lambdas_trunc += lambdas[i]
        n_L += 1
        if sum_lambdas_trunc/sum_lambdas >= (1 - eps):
            break
    
    # Truncating according to n_L
    lambdas_trunc = lambdas[0:n_L]
    
    # Constructiong the reduced POD base V
    V = np.zeros((n_x, n_y, n_L))
    for i in range(n_L):
        V[:, :, i] = U_h.dot(Z[:, i]) / np.sqrt(lambdas_trunc[i])
    
    if verbose:
        # Number of solutions and Stopping parameter
        print("n_s: ", n_s)
        print("eps: ", eps)
        
        print(f"Elapsed time is {time.time() - start_time} seconds.")
        print(f"L = {n_L}")
    
    if write_file:
        name = f"shek_Pod_bases_lhs_nxy_{n_e}_ns_{n_s}_epsi_{eps}.txt"
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

