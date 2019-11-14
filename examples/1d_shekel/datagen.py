"""HiFi testing data generation for 1D Shekel Equation."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.plotting import figsize, openPdfGraph
from podnn.testgenerator import TestGenerator, X_FILE, U_MEAN_FILE, U_STD_FILE

from hyperparams import HP


# HiFi sampling size
n_s = int(1e6)


def u(X, _, mu):
    """The 1D-Shekel function."""
    x = X[0]
    bet = mu[:10]
    gam = mu[10:]

    u_sum = np.zeros_like(x)
    for i in range(len(bet)):
        i_sum = (x - gam[i])**2
        u_sum += 1 / (bet[i] + i_sum)

    return u_sum


class ShekelTestGenerator(TestGenerator):
    def plot(self):
        dirname = "data"
        print(f"Reading data to {dirname}")
        x = np.load(os.path.join(dirname, X_FILE)).reshape((300,))
        u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
        u_std = np.load(os.path.join(dirname, U_STD_FILE))
        # Keeping just the first coordinate (1D)
        u_mean = u_mean[0, :]
        u_std = u_std[0, :]

        fig = plt.figure(figsize=figsize(1, 2, 2.0))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_mean.plot(x, u_mean)
        ax_mean.set_title(r"Mean of $u_h(x, \gamma, \beta)$")
        ax_mean.set_xlabel("$x$")
        ax_std = fig.add_subplot(1, 2, 2)
        ax_std.plot(x, u_std)
        ax_std.set_title(r"Standard deviation of $u_h(x, \gamma, \beta)$")
        ax_std.set_xlabel("$x$")

        PLOT_FILE = os.path.join(dirname, "plot")
        plt.savefig(PLOT_FILE + ".pdf")
        openPdfGraph(PLOT_FILE)



def generate_test_dataset():
    tg = ShekelTestGenerator(u, HP["n_v"], HP["n_x"])
    tg.generate(n_s, HP["mu_min"], HP["mu_max"], HP["x_min"], HP["x_max"],
                parallel=True)
    return tg


if __name__ == "__main__":
    testgen = generate_test_dataset()
    testgen.plot()
