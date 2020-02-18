"""Various utilities functions."""

import os
import argparse
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


def check_distributed_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--distributed", action="store_true", default=False)
    pa.add_argument("--models", type=int, default=1)
    args = pa.parse_args()
    return args.distributed, args.models

def clean_dir(dirname):
    for root, dirs, files in os.walk(dirname):
        for name in files:
            if not name.endswith(".gitignore"):
                os.remove(os.path.join(root, name))