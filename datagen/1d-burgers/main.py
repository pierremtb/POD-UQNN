import sys
import os
import numpy as np
from tqdm import tqdm
from pyDOE import lhs

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
from plots import plot_results
from hyperparams import hp

sys.path.append(os.path.join(eqnPath, "burgersutils"))
from burgers import burgers_viscous_time_exact1 as burgers_u

from names import X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE

# Hyperparameters
n_x = hp["n_x"]
n_t = hp["n_t"]
x_min = hp["x_min"]
x_max = hp["x_max"]
t_min = hp["t_min"]
t_max = hp["t_max"]
mu_mean = hp["mu_mean"]

# HiFi sampling size
n_s = int(1e3)

# Static data
x = np.linspace(x_min, x_max, n_x)
t = np.linspace(t_min, t_max, n_t)
XT, TT = np.meshgrid(x, t)
X = XT.T
T = TT.T

n_h = n_x
n_d = 1 + 1
lb = mu_mean * (1 - np.sqrt(3)/10)
ub = mu_mean * (1 + np.sqrt(3)/10)

# The sum and sum of squares recipient vectors
U_tot = np.zeros((n_x, n_t))
U_tot_sq = np.zeros((n_x, n_t))

# Going through the snapshots one by one without saving them
for i in tqdm(range(n_s)):
    # Computing one snapshot
    X_mu = lhs(1, 1).T
    mu_lhs = lb + (ub - lb)*X_mu
    U = burgers_u(mu_lhs[0, 0], n_x, x, n_t, t)

    # Building the sum and the sum of squaes
    U_tot += U
    U_tot_sq += U**2

# Recreating the mean and the std
U_test_mean = U_tot / n_s
U_test_std = np.sqrt((n_s*U_tot_sq - U_tot**2) / (n_s*(n_s - 1)))

dirname = os.path.join(eqnPath, "data")
print(f"Saving data to {dirname}")
np.save(os.path.join(dirname, X_FILE), X)
np.save(os.path.join(dirname, T_FILE), T)
np.save(os.path.join(dirname, U_MEAN_FILE), U_test_mean)
np.save(os.path.join(dirname, U_STD_FILE), U_test_std)
