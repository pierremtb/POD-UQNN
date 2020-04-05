"""Various utilities functions."""

import numpy as np
from .acceleration import lhs


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


def sample_mu(n_s, mu_min, mu_max, indices=None):
    """Return a LHS sampling between mu_min and mu_max of size n_s."""
    if indices is not None:
        mu = np.linspace(mu_min, mu_max, n_s)[indices]
        return mu
    X_lhs = lhs(n_s, mu_min.shape[0]).T
    mu_lhs = mu_min + (mu_max - mu_min)*X_lhs
    return mu_lhs
