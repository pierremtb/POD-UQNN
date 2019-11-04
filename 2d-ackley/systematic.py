import sys
import os
import json
import numpy as np
import tensorflow as tf


eqnPath = "1d-shekel"
from main import main
from hyperparams import hp


if __name__ == "__main__":
    # Study parameters
    list_samples = [50, 200, 600, 1000, 2000]
    list_epochs = [1000, 5000, 20000, 50000, 100000]

    # Results containers
    errors_test_mean = np.zeros((len(list_samples), len(list_epochs)))
    errors_test_std = np.zeros((len(list_samples), len(list_epochs)))

    # Running
    for i_n_s, n_s in enumerate(list_samples):
        print(f"For n_s={n_s}...")
        for i_tf_epochs, tf_epochs in enumerate(list_epochs):
            print(f"For tf_epochs={tf_epochs}")
            hp["n_s"] = n_s
            hp["epochs"] = tf_epochs
            error_test_mean, error_test_std = main(hp)

    # Saving the results
    np.savetxt(os.path.join(eqnPath, "results", "systematic", "n_s.csv"),
               hp["n_s"])
    np.savetxt(os.path.join(eqnPath, "results", "systematic", "tf_epochs.csv"),
               hp["tf_epochs"])
    np.savetxt(os.path.join(eqnPath, "results", "systematic", "err_t_mean.csv"),
               errors_test_mean)
    np.savetxt(os.path.join(eqnPath, "results", "systematic", "err_t_std.csv"),
               errors_test_std)
