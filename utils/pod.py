import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm


def get_pod_bases(U_h, eps=1e-10):
    start_time = time.time()
    
    # Number of DOFs
    n_h = U_h.shape[0]
    # Number of snapshots n_t x Number of space nodes (n_x * n_y * ...)
    nn_t = U_h.shape[1]
    
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
    for i in range(nn_t):
        sum_lambdas_trunc += lambdas[i]
        n_L += 1
        if sum_lambdas_trunc/sum_lambdas >= (1 - eps):
            break
    
    # Truncating according to n_L
    lambdas_trunc = lambdas[0:n_L]
    
    # Constructiong the reduced POD base V
    V = np.zeros((n_h, n_L))
    for i in range(n_L):
        V[:, i] = U_h.dot(Z[:, i]) / np.sqrt(lambdas_trunc[i])
    
    return V

