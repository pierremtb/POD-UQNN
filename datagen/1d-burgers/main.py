import sys
import os
import numpy as np
from tqdm import tqdm
from pyDOE import lhs
import json

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
from plots import plot_results
from hyperparams import hp

sys.path.append(os.path.join(eqnPath, "burgersutils"))
from burgers import burgers_viscous_time_exact1 as burgers_u

sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE, HP_FILE

# HiFi sampling size
n_s = int(1e2)

# Static data
x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
XT, TT = np.meshgrid(x, t)
X = XT.T
T = TT.T

n_h = hp["n_x"]
n_d = 1 + 1
lb = hp["mu_min"]
ub = hp["mu_max"]

# The sum and sum of squares recipient vectors
U_tot = np.zeros((hp["n_x"], hp["n_t"]))
U_tot_sq = np.zeros((hp["n_x"], hp["n_t"]))

# Going through the snapshots one by one without saving them
for i in tqdm(range(n_s)):
    # Computing one snapshot
    X_mu = lhs(1, 1).T
    mu_lhs = lb + (ub - lb)*X_mu
    U = burgers_u(mu_lhs[0, 0], x.shape[0], x, t.shape[0], t)

    # Building the sum and the sum of squaes
    U_tot += U
    U_tot_sq += U**2

# Recreating the mean and the std
U_test_mean = U_tot / n_s
U_test_std = np.sqrt((n_s*U_tot_sq - U_tot**2) / (n_s*(n_s - 1)))
# Making sure the std has non NaNs
U_test_std = np.nan_to_num(U_test_std)

dirname = os.path.join(eqnPath, "data")
print(f"Saving data to {dirname}")
np.save(os.path.join(dirname, X_FILE), X)
np.save(os.path.join(dirname, T_FILE), T)
np.save(os.path.join(dirname, U_MEAN_FILE), U_test_mean)
np.save(os.path.join(dirname, U_STD_FILE), U_test_std)

# Store the HiFi hyperparams
hp_hifi = {}
hp_hifi["n_x"] = hp["n_x"]
hp_hifi["x_min"] = hp["x_min"]
hp_hifi["x_max"] = hp["x_max"]
hp_hifi["n_t"] = hp["n_t"]
hp_hifi["t_min"] = hp["t_min"]
hp_hifi["t_max"] = hp["t_max"]
hp_hifi["mu_min"] = hp["mu_min"]
hp_hifi["mu_max"] = hp["mu_max"]
hp_hifi["n_s"] = n_s
with open(os.path.join(dirname, HP_FILE), "w") as f:
    json.dump(hp_hifi, f)
