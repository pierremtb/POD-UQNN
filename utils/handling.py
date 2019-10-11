import numpy as np


def scarcify(X, u, N):
    idx = np.random.choice(X.shape[0], N, replace=False)
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[idx, :], u[idx, :], X[mask, :], u[mask, :]


def restruct(U_h, n_x, n_y, n_s):
    U_h_struct = np.zeros((n_x, n_y, n_s))
    idx = np.arange(n_s) * n_x * n_y
    for i in range(n_x):
        U_h_struct[i, :] = U_h[:, idx + i]
    return U_h_struct
