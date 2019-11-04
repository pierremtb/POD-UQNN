import numpy as np
import sys
import os
from tqdm import tqdm
import pickle
from pyDOE import lhs
import matplotlib.pyplot as plt
from deap.benchmarks import shekel

eqnPath = "1d-shekel"
sys.path.append(eqnPath)

sys.path.append("utils")
from plotting import figsize
from hyperparams import hp
from pod import get_pod_bases
from testgenerator import TestGenerator, X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE


def u(X, t, mu):
    x = X[0]
    bet, gam = mu[:10], mu[10:]
    return -shekel(x[None, :], gam.reshape((10, 1)), bet.reshape((10, 1)))[0]


class ShekelTestGenerator(TestGenerator):
  def plot(self):
      dirname = os.path.join(eqnPath, "data")
      print(f"Reading data to {dirname}")
      x = np.load(os.path.join(dirname, X_FILE)).reshape((300,))
      u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
      u_std = np.load(os.path.join(dirname, U_STD_FILE))
      
      fig = plt.figure(figsize=figsize(2, 1))
      ax_mean = fig.add_subplot(1, 2, 1)
      ax_mean.plot(x, u_mean)
      ax_mean.set_title(r"Mean of $u_h(x, \gamma, \beta)$")
      ax_mean.set_xlabel("$x$")
      ax_std = fig.add_subplot(1, 2, 2)
      ax_std.plot(x, u_std)
      ax_std.set_title(r"Standard deviation of $u_h(x, \gamma, \beta)$")
      ax_std.set_xlabel("$x$")
      plt.show()



def generate_test_dataset():
  testgen = ShekelTestGenerator(eqnPath, u, hp["n_v"], hp["n_x"])
  testgen.generate(int(1e2), hp["mu_min"], hp["mu_max"], hp["x_min"], hp["x_max"])
  return testgen


if __name__ == "__main__":
    testgen = generate_test_dataset()
    testgen.plot()
