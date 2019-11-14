"""HiFi testing data generation for 1D time-dep Burgers eq."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join("..", ".."))
from podnn.plotting import figsize, openPdfGraph
from podnn.testgenerator import TestGenerator, X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE

from hyperparams import HP

sys.path.append("burgersutils")
from burgers import burgers_viscous_time_exact1 as burgers_exact


# HiFi sampling size
# n_s = int(1e6)
n_s = int(1e2)


# The solution function
def u(X, t, mu):
    x = X[0]
    return burgers_exact(mu, x.shape[0], x, 1, [t]).T


class BurgersTestGenerator(TestGenerator):
    def plot(self):
        dirname = os.path.join("data")
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
        fig = plt.figure(figsize=figsize(1, 2, 2.0))
        ax_mean = fig.add_subplot(121, projection="3d")
        ax_mean.plot_surface(X, T, u_mean)
        ax_mean.set_title(r"Mean of $u_h(x, \gamma, \beta)$")
        ax_mean.set_xlabel("$x$")
        ax_std = fig.add_subplot(122, projection="3d")
        ax_std.plot_surface(X, T, u_std)
        ax_std.set_title(r"Standard deviation of $u_h(x, \gamma, \beta)$")
        ax_std.set_xlabel("$x$")

        PLOT_FILE = os.path.join(dirname, "plot")
        plt.savefig(PLOT_FILE + ".pdf")
        openPdfGraph(PLOT_FILE)


def generate_test_dataset():
    tg = BurgersTestGenerator(u, HP["n_v"], HP["n_x"], n_t=HP["n_t"])
    tg.generate(n_s, HP["mu_min"], HP["mu_max"], HP["x_min"], HP["x_max"],
                t_min=HP["t_min"], t_max=HP["t_max"])
    return tg


if __name__ == "__main__":
    testgen = generate_test_dataset()
    testgen.plot()