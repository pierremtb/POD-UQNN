import numpy as np


def pack_layers(i, hiddens, o):
    layers = []
    layers.append(i)
    for h in hiddens:
        layers.append(h)
    layers.append(o)
    return layers


def scarcify(X, u, N):
    idx = np.random.choice(X.shape[0], N, replace=False)
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[idx, :], u[idx, :], X[mask, :], u[mask, :]


def restruct(U, n_x, n_y, n_s):
    U_struct = np.zeros((n_x, n_y, n_s))
    idx = np.arange(n_s) * n_x * n_y
    for i in range(n_x):
        U_struct[i, :] = U[:, idx + i]
    return U_struct
