import sys
import os
import numpy as np
from tqdm import tqdm
from pyDOE import lhs
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

eqnPath = "1dt-burgers2"
sys.path.append(eqnPath)
from plots import plot_results
from hyperparams import hp

sys.path.append("utils")
from plotting import figsize
from testgenerator import TestGenerator, X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE


# HiFi sampling size
n_s = int(1e3)


# The solution function
def u(X, t, mu):
    x = X[0]

    if t == 1.:
        return x / (1 + np.exp(1/(4*mu)*(x**2 - 1/4)))

    t0 = np.exp(1 / (8*mu))
    return (x/t) / (1 + np.sqrt(t/t0)*np.exp(x**2/(4*mu*t)))


class BurgersTestGenerator(TestGenerator):
  def plot(self):
    dirname = os.path.join(eqnPath, "data")
    print(f"Reading data to {dirname}")

    # Loading space
    X = np.load(os.path.join(dirname, X_FILE))
    t = np.load(os.path.join(dirname, T_FILE))
    # Keeping the first coordinate, and meshing with t
    x = X[0]
    Xt, Tt = np.meshgrid(x, t)
    X, T = Xt.T, Tt.T

    # Loading solution and keeping its first coordinate (n_v == 1)
    u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    u_std = np.load(os.path.join(dirname, U_STD_FILE))
    u_mean = u_mean[0, :, :]
    u_std = u_std[0, :, :]

    # Plotting
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
    # plt.plot(X[:, 0], u_mean[:, 25])
    # plt.plot(X[:, 0], u_mean[:, 50])
    # plt.plot(X[:, 0], u_mean[:, 75])
    # plt.show()


def generate_test_dataset():
  testgen = BurgersTestGenerator(eqnPath, u, hp["n_v"], hp["n_x"], n_t=hp["n_t"])
  testgen.generate(n_s, hp["mu_min"], hp["mu_max"], hp["x_min"], hp["x_max"],
                   t_min=hp["t_min"], t_max=hp["t_max"])
  return testgen


if __name__ == "__main__":
    testgen = generate_test_dataset()
    testgen.plot()
