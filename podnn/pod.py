import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from numba import njit, prange


@njit(["f8[:, :](f8[:, :], f8, b1)"], parallel=True)
def perform_pod(U, eps, verbose=True):
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
    n_L = 0
    sum_lambdas_trunc = 0.
    for i in prange(n_st):
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
    # for i in tqdm(prange(n_L), disable=(not verbose)):
    for i in prange(n_L):
        Z_i = np.ascontiguousarray(Z[:, i])
        V[:, i] = U.dot(Z_i) / np.sqrt(lambdas_trunc[i])
    
    return np.ascontiguousarray(V)


@njit("f8[:, :](f8[:, :, :], f8, f8)", parallel=True)
def perform_fast_pod(U, eps, eps_init):
    """Two-step version of POD algorithm."""
    print("Performing initial time-trajectory POD")
    # Number of DOFs
    n_h = U.shape[0]
    # Number of snapshots n_s x Number of space nodes (n_x * n_y * ...)
    n_s = U.shape[-1]

    # Init at the max it can be, n_t
    n_L_init = U.shape[1]
    
    T = np.zeros((n_h, n_L_init, n_s))
    for k in range(n_s):
        U_k = U[:, :, k]
        T_k = perform_pod(U_k, eps_init, False)
        n_L_k = T_k.shape[1]
        if n_L_k < n_L_init:
            n_L_init = n_L_k
        for i in range(n_L_init):
            T[:, i, k] = T_k[:, i]
        # T[:, :n_L_init, k] = T_k[:, :n_L_init]

    # Cropping the results accordingly and stacking
    T = np.ascontiguousarray(T[:, :n_L_init, :])
    # Reshaping the 3d-mat into a 2d-mat
    U_f = np.reshape(T, (n_h, n_s*n_L_init))
    print("Performing SVD")
    return perform_pod(U_f, eps, True)
