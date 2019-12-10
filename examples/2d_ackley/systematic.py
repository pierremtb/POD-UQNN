import os
import numpy as np

from main import main
from hyperparams import HP


if __name__ == "__main__":
    # Study parameters
    list_samples = [100, 300, 600, 1000]
    list_epochs = [30000, 50000, 75000, 150000]

    # Results containers
    errors_test_mean = np.zeros((len(list_samples), len(list_epochs)))
    errors_test_std = np.zeros((len(list_samples), len(list_epochs)))

    # Running
    for i_n, n_s in enumerate(list_samples):
        print(f"For n_s={n_s}")
        for i_e, tf_epochs in enumerate(list_epochs):
            print(f"For tf_epochs={tf_epochs}")
            HP["n_s"] = n_s
            HP["epochs"] = tf_epochs
            errors_test_mean[i_n, i_e], errors_test_std[i_n, i_e] = main(HP, no_plot=True)

    # Saving the results
    np.savetxt(os.path.join("results", "systematic", "n_s.csv"),
               list_samples)
    np.savetxt(os.path.join("results", "systematic", "tf_epochs.csv"),
               list_epochs)
    np.savetxt(os.path.join("results", "systematic", "err_t_mean.csv"),
               errors_test_mean)
    np.savetxt(os.path.join("results", "systematic", "err_t_std.csv"),
               errors_test_std)
