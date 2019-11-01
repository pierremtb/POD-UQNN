import sys
import os
import numpy as np
from tqdm import tqdm
from pyDOE import lhs
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

eqnPath = "1d-burgers2"
sys.path.append(eqnPath)
from plots import plot_results
from hyperparams import hp

sys.path.append("utils")
from plotting import figsize

# NAMES
X_FILE = "x.npy"
T_FILE = "t.npy"
U_MEAN_FILE = "u_mean.npy"
U_STD_FILE = "u_std.npy"
HP_FILE = "hp.json"

# HiFi sampling size
# n_s = int(10)
n_s = int(1e3)


# The solution function
def u(X, t, mu):
    x = X[0]

    if t == 1.:
        return x / (1 + np.exp(1/(4*mu)*(x**2 - 1/4)))

    t0 = np.exp(1 / (8*mu))
    return (x/t) / (1 + np.sqrt(t/t0)*np.exp(x**2/(4*mu*t)))


def generate(n_s):
    # Static data
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
    XT, TT = np.meshgrid(x, t)
    X = XT.T
    T = TT.T

    n_h = hp["n_x"]
    n_d = 1 + 1
    lb = hp["mu_min"][0]
    ub = hp["mu_max"][0]

    # The sum and sum of squares recipient vectors
    U_tot = np.zeros((hp["n_x"], hp["n_t"]))
    U_tot_sq = np.zeros((hp["n_x"], hp["n_t"]))

    # Going through the snapshots one by one without saving them
    for i in tqdm(range(n_s)):
        # Computing one snapshot
        X_mu = lhs(1, 1).T
        mu_lhs = lb + (ub - lb)*X_mu
        U = u(x, t, mu_lhs[0, 0])

        # Building the sum and the sum of squaes
        U_tot += U
        U_tot_sq += U**2

    # Recreating the mean and the std
    U_test_mean = U_tot / n_s
    U_test_std = np.sqrt((n_s*U_tot_sq - U_tot**2) / (n_s*(n_s - 1)))
    # Making sure the std has non NaNs
    U_test_std = np.nan_to_num(U_test_std)

    dirname = os.path.join(eqnPath, "data")
    print(f"Saving data to {dirname}")
    np.save(os.path.join(dirname, X_FILE), X)
    np.save(os.path.join(dirname, T_FILE), T)
    np.save(os.path.join(dirname, U_MEAN_FILE), U_test_mean)
    np.save(os.path.join(dirname, U_STD_FILE), U_test_std)

    # Store the HiFi hyperparams
    hp_hifi = {}
    hp_hifi["n_x"] = hp["n_x"]
    hp_hifi["x_min"] = hp["x_min"]
    hp_hifi["x_max"] = hp["x_max"]
    hp_hifi["n_t"] = hp["n_t"]
    hp_hifi["t_min"] = hp["t_min"]
    hp_hifi["t_max"] = hp["t_max"]
    hp_hifi["mu_min"] = hp["mu_min"]
    hp_hifi["mu_max"] = hp["mu_max"]
    hp_hifi["n_s"] = n_s
    with open(os.path.join(dirname, HP_FILE), "w") as f:
        json.dump(hp_hifi, f)


def plot():
    dirname = os.path.join(eqnPath, "data")
    print(f"Reading data to {dirname}")
    X = np.load(os.path.join(dirname, X_FILE))
    T = np.load(os.path.join(dirname, T_FILE))
    u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    u_std = np.load(os.path.join(dirname, U_STD_FILE))

    fig = plt.figure(figsize=figsize(2, 1))
    ax_mean = fig.add_subplot(121, projection="3d")
    ax_mean.plot_surface(X, T, u_mean)
    ax_mean.set_title(r"Mean of $u_h(x, \gamma, \beta)$")
    ax_mean.set_xlabel("$x$")
    ax_std = fig.add_subplot(122, projection="3d")
    ax_std.plot_surface(X, T, u_std)
    ax_std.set_title(r"Standard deviation of $u_h(x, \gamma, \beta)$")
    ax_std.set_xlabel("$x$")
    plt.show()

    plt.plot(X[:, 0], u_mean[:, 25])
    plt.plot(X[:, 0], u_mean[:, 50])
    plt.plot(X[:, 0], u_mean[:, 75])
    plt.show()


if __name__ == "__main__":
    generate(n_s)
    plot()
