"""Various utilities functions."""

import numpy as np


def pack_layers(i, hiddens, o):
    """Create the full NN topology from input size, hidden layers, and output."""

    layers = []
    layers.append(i)
    for h in hiddens:
        layers.append(h)
    layers.append(o)
    return layers


def scarcify(X, u, N):
    """Randomly split a dataset into train-val subsets."""

    idx = np.random.choice(X.shape[0], N, replace=False)
    mask = np.ones(X.shape[0], bool)
    mask[idx] = False
    return X[idx, :], u[idx, :], X[mask, :], u[mask, :]

def split_dataset(X_v, v, test_size, idx_only=False):
    """Randomly splitting the dataset (X_v, v)."""
    indices = np.random.permutation(X_v.shape[0])
    limit = np.floor(X_v.shape[0] * (1. - test_size)).astype(int)
    if idx_only:
        return indices[:limit].tolist(), indices[limit:].tolist()
    train_idx, tst_idx = indices[:limit], indices[limit:]
    return X_v[train_idx], X_v[tst_idx], v[train_idx], v[tst_idx]
