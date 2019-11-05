import numpy as np
import sys
import os
from tqdm import tqdm
import pickle
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

eqnPath = "2d-ackley"
sys.path.append(eqnPath)

sys.path.append("utils")
from plotting import figsize
from hyperparams import hp
from pod import get_pod_bases
from testgenerator import TestGenerator, X_FILE, \
        T_FILE, U_MEAN_FILE, U_STD_FILE


# HiFi sampling size
n_s = int(1e3)


def u(X, t, mu):
    x = X[0]
    y = X[1]
    return - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
           - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
           + 20 + np.exp(1)


class AckleyTestGenerator(TestGenerator):
  def plot(self):
    dirname = os.path.join(eqnPath, "data")
    print(f"Reading data to {dirname}")
    X = np.load(os.path.join(dirname, X_FILE))
    x, y = X[0], X[1]
    u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    u_std = np.load(os.path.join(dirname, U_STD_FILE))

    # Keepinp the first coordinate
    u_mean = u_mean[0, :, :]
    u_std = u_std[0, :, :]

    fig = plt.figure(figsize=figsize(2, 1))
    ax_mean = fig.add_subplot(121, projection="3d")
    ax_mean.plot_surface(x, y, u_mean)
    ax_mean.set_title(r"Mean of $u_h(x, \gamma, \beta)$")
    ax_mean.set_xlabel("$x$")
    ax_std = fig.add_subplot(122, projection="3d")
    ax_std.plot_surface(x, y, u_std)
    ax_std.set_title(r"Standard deviation of $u_h(x, \gamma, \beta)$")
    ax_std.set_xlabel("$x$")
    plt.show()


def generate_test_dataset():
  testgen = AckleyTestGenerator(eqnPath, u, hp["n_v"],
                                hp["n_x"], hp["n_y"])
  testgen.generate(int(1e2), hp["mu_min"], hp["mu_max"],
                   hp["x_min"], hp["x_max"],
                   hp["y_min"], hp["y_max"])
  return testgen


if __name__ == "__main__":
    testgen = generate_test_dataset()
    testgen.plot()

