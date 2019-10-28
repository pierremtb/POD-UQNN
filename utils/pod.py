import numpy as np
import tensorflow as tf
from tqdm import tqdm


def get_pod_bases(U, eps=1e-10, eps_init_step=None): 
    if eps_init_step is not None:
        eps_init_step = 1e-8
        print("Performing initial time-trajectory POD")
        # Number of DOFs
        n_h = U.shape[0]
        # Number of snapshots n_s x Number of space nodes (n_x * n_y * ...)
        n_s = U.shape[-1]
        T_raw = []

        # Init at the max it can be, n_t
        n_L_init = U.shape[1]
        T = np.zeros((n_h, n_L_init, n_s))
        for k in range(n_s):
            T_k = perform_pod(U[:, :, k], eps_init_step, verbose=False)
            if T_k.shape[1] < n_L_init:
                n_L_init = T_k.shape[1]
            T[:, :n_L_init, k] = T_k[:, :n_L_init]

        # Cropping the results accordingly and stacking
        T = T[:, :n_L_init, :]
        # Reshaping the 3d-mat into a 2d-mat
        U = np.reshape(T, (n_h, n_s*n_L_init))
    
    print("Performing SVD...")
    return perform_pod(U, eps)

def perform_pod(U, eps, verbose=True):
    # Number of DOFs
    n_h = U.shape[0]
    # Number of snapshots n_s x Number of space nodes (n_x * n_y * ...)
    nn_s = U.shape[1]

    W, D, ZT = np.linalg.svd(U, full_matrices=False)

    
    # Getting MATLAB-like orientation
    Z = ZT.T
    
    # Storing eigenvalues and their sum
    lambdas = D**2
    sum_lambdas = np.sum(lambdas)
    
    # Finding n_L
    n_L = 0
    sum_lambdas_trunc = 0.
    for i in range(nn_s):
        sum_lambdas_trunc += lambdas[i]
        n_L += 1
        if sum_lambdas_trunc/sum_lambdas >= (1 - eps):
            break
    
    # Truncating according to n_L
    lambdas_trunc = lambdas[0:n_L]
   
    if verbose:
        print("Contructing the reduced bases V...")
    V = np.zeros((n_h, n_L))
    for i in tqdm(range(n_L), disable=(not verbose)):
        V[:, i] = U.dot(Z[:, i]) / np.sqrt(lambdas_trunc[i])
    
    return V

