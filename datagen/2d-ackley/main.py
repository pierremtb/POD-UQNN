import sys
import os
import numpy as np
from tqdm import tqdm
from pyDOE import lhs

eqnPath = "2d-ackley"
sys.path.append(eqnPath)
sys.path.append("utils")
from ackleyutils import plot_results, u_h

from names import X_FILE, Y_FILE, U_MEAN_FILE, U_STD_FILE

# Hyperparameters
n_x = 400
n_y = 400
n_s = int(1e5)
x_min = -5.
x_max = 5.
y_min = -5.
y_max = 5.

# Static data
x = np.linspace(x_min, x_max, n_x)
y = np.linspace(x_min, y_max, n_y)
X, Y = np.meshgrid(x, y)
n_h = n_x * n_y
mu = np.array([0., 0., 0.])
n_d = 2 + mu.shape[0]
lb = -1. * np.ones_like(mu)
ub = +1. * np.ones_like(mu)

# The sum and sum of squares recipient vectors
U_tot = np.zeros((n_x*n_y, 1))
U_tot_sq = np.zeros((n_x*n_y, 1))

# Going through the snapshots one by one without saving them
for i in tqdm(range(n_s)):
    # Computing one snapshot
    X_mu = lhs(1, mu.shape[0]).T
    mu_lhs = lb + (ub - lb)*X_mu
    U = np.reshape(u_h(X, Y, mu_lhs[0, :]), (n_x * n_y, 1))

    # Building the sum and the sum of squaes
    U_tot += U
    U_tot_sq += U**2

# Recreating the mean and the std
U_mean = U_tot / n_s
U_std = np.sqrt((n_s*U_tot_sq - U_tot**2) / (n_s*(n_s - 1)))

# Reshaping into a 2D-valued solution
U_test_mean = np.reshape(U_mean, (n_x, n_y))
U_test_std = np.reshape(U_std, (n_x, n_y))

dirname = os.path.join(eqnPath, "data")
print(f"Saving data to {dirname}")
np.save(os.path.join(dirname, X_FILE), X)
np.save(os.path.join(dirname, Y_FILE), Y)
np.save(os.path.join(dirname, U_MEAN_FILE), U_test_mean)
np.save(os.path.join(dirname, U_STD_FILE), U_test_std)
