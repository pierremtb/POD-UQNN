import numpy as np
import os
import sys


arr_n_s = np.loadtxt(os.path.join("results", "systematic", "n_s.csv"))
arr_tf_epochs = np.loadtxt(os.path.join("results", "systematic", "tf_epochs.csv"))
errors_test_mean = np.loadtxt(os.path.join("results", "systematic", "err_t_mean.csv"))
errors_test_std = np.loadtxt(os.path.join("results", "systematic", "err_t_std.csv"))
errors = np.stack((errors_test_mean, errors_test_std), axis=2)

import matplotlib.pyplot as plt
sys.path.append(os.path.join("..", ".."))
from poduqnn.plotting import figsize, savefig

fig = plt.figure(figsize=figsize(1, 2, scale=2.))
colors = ["k", "b", "g"]
for i in range(len(colors)):
    plt.plot(arr_n_s, 100*errors_test_mean[:, i], f"{colors[i]}--", label=f"$N_e={int(arr_tf_epochs[i])}$, val")
    plt.plot(arr_n_s, 100*errors_test_std[:, i], f"{colors[i]}-", label=f"$N_e={int(arr_tf_epochs[i])}$, tst")
plt.xlabel("$S$")
plt.ylabel(r"RE [\%]")
plt.legend()
savefig(os.path.join("cache", "podnn-shekel-systematic"))