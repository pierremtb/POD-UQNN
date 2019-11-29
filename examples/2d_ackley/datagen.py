"""HiFi testing data generation for 2D Ackley Equation."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join("..", ".."))
from podnn.plotting import figsize, openPdfGraph
from podnn.testgenerator import TestGenerator, X_FILE, U_MEAN_FILE, U_STD_FILE

from hyperparams import HP


# HiFi sampling size
n_s = HP["n_s_hifi"]


def u(X, _, mu):
    x = X[0]
    y = X[1]
    u_0 = - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
          - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
          + 20 + np.exp(1)
    return u_0.reshape((1, u_0.shape[0]))


class AckleyTestGenerator(TestGenerator):
    def plot(self):
        dirname = os.path.join("data")
        print(f"Reading data to {dirname}")
        X = np.load(os.path.join(dirname, X_FILE))
        x, y = X[0], X[1]
        u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
        u_std = np.load(os.path.join(dirname, U_STD_FILE))

        # Keepinp the first coordinate
        u_mean = u_mean.reshape(self.get_u_tuple())[0, :, :]
        u_std = u_std.reshape(self.get_u_tuple())[0, :, :]

        fig = plt.figure(figsize=figsize(1, 2, 2.0))
        ax_mean = fig.add_subplot(121, projection="3d")
        ax_mean.plot_surface(x, y, u_mean)
        ax_mean.set_title(r"Mean of $u_h(x, \gamma, \beta)$")
        ax_mean.set_xlabel("$x$")
        ax_std = fig.add_subplot(122, projection="3d")
        ax_std.plot_surface(x, y, u_std)
        ax_std.set_title(r"Standard deviation of $u_h(x, \gamma, \beta)$")
        ax_std.set_xlabel("$x$")

        PLOT_FILE = os.path.join(dirname, "plot")
        plt.savefig(PLOT_FILE + ".pdf")
        openPdfGraph(PLOT_FILE)


def generate_test_dataset():
    tg = AckleyTestGenerator(u, HP["n_v"], HP["n_x"], HP["n_y"])
    tg.generate(n_s, HP["mu_min"], HP["mu_max"],
                HP["x_min"], HP["x_max"],
                HP["y_min"], HP["y_max"],
                parallel=True)
    return tg


if __name__ == "__main__":
    testgen = generate_test_dataset()
    testgen.plot()
