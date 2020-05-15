"""Module to handle Proper Orthogonal Decomposition tasks."""
import numpy as np
from numba import njit


@njit(parallel=False)
def perform_pod(U, eps=0., n_L=0, verbose=True):
    """POD algorithmm."""
    # Number of DOFs
    n_h = U.shape[0]

    # Number of snapshots n_s x Number of time steps
    n_st = U.shape[1]

    # SVD algoritm call
    _, D, ZT = np.linalg.svd(U, full_matrices=False)

    # Getting MATLAB-like orientation
    Z = ZT.T
 
    # Storing eigenvalues and their sum
    lambdas = D**2
    sum_lambdas = np.sum(lambdas)
  
    # Finding n_L
    if n_L == 0:
        sum_lambdas_trunc = 0.
        for i in range(n_st):
            sum_lambdas_trunc += lambdas[i]
            n_L += 1
            if sum_lambdas_trunc/sum_lambdas >= (1 - eps):
                break
 
    # Truncating according to n_L
    lambdas_trunc = lambdas[0:n_L]
  
    if verbose:
        print("Contructing the reduced bases V")

    U = np.ascontiguousarray(U)

    V = np.zeros((n_h, n_L))
    for i in range(n_L):
        Z_i = np.ascontiguousarray(Z[:, i])
        V[:, i] = U.dot(Z_i) / np.sqrt(lambdas_trunc[i])

    return np.ascontiguousarray(V)


@njit(parallel=False)
def perform_fast_pod(U, eps, eps_init):
    """Two-step version of POD algorithm."""
    print("Performing initial time-trajectory POD")
    # Number of snapshots n_s x Number of space nodes (n_x * n_y * ...)
    n_s = U.shape[-1]

    T_list = []
    for k in range(n_s):
        U_k = U[:, :, k]
        # Retrieving each time-trajectory
        T_k = perform_pod(U_k, eps=eps_init, n_L=0, verbose=False)
        T_list.append(T_k)

    # Reshaping the 3d-mat into a 2d-mat
    U_f = np.concatenate(T_list, axis=1)

    print("Performing SVD")
    return perform_pod(U_f, eps=eps, n_L=0, verbose=True)
