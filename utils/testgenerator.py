import numpy as np
import sys
import os
from tqdm import tqdm
import pickle
import json
from pyDOE import lhs

X_FILE = "X.npy"
T_FILE = "t.npy"
U_MEAN_FILE = "u_mean.npy"
U_STD_FILE = "u_std.npy"
HP_FILE = "hp.json"


class TestGenerator(object):
    def __init__(self, eqnPath, u, n_v, n_x, n_y=0, n_z=0, n_t=0):
        self.eqnPath = eqnPath
        self.u = u
        self.n_v = n_v
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.n_t = n_t
        self.has_y = n_y > 0
        self.has_z = n_z > 0
        self.has_t = n_t > 0

    def plot(self):
        raise NotImplementedError

    def generate(self, n_s, mu_min, mu_max, x_min, x_max,
                 y_min=0, y_max=0, z_min=0, z_max=0,
                 t_min=0, t_max=0):
        # Static data
        # TODO: y and z
        x = np.linspace(x_min, x_max, self.n_x)
        if self.has_t:
            t = np.linspace(t_min, t_max, self.n_t)
        X = [x]

        n_h = self.n_v * self.n_x
        n_p = mu_min.shape[0]

        n_d = n_p
        if self.has_t:
            n_d += 1

        lb = mu_min
        ub = mu_max

        # The sum and sum of squares recipient vectors
        if self.has_t > 0:
            U_tot = np.zeros((n_h, self.n_t))
            U_tot_sq = np.zeros((n_h, self.n_t))
        else:
            U_tot = np.zeros((n_h,))
            U_tot_sq = np.zeros((n_h,))

        # Going through the snapshots one by one without saving them
        for i in tqdm(range(n_s)):
            # Computing one snapshot
            X_mu = lhs(n_s, n_p).T
            mu_lhs = lb + (ub - lb)*X_mu
            U = np.zeros_like(U_tot)
            if self.has_t > 0:
                for i in range(self.n_t):
                    U[:, i] = self.u(X, t[i], mu_lhs[i, :])
            else:
                U = self.u(X, 0, mu_lhs[i, :])

            # Building the sum and the sum of squaes
            U_tot += U
            U_tot_sq += U**2

        # Recreating the mean and the std
        U_test_mean = U_tot / n_s
        U_test_std = np.sqrt((n_s*U_tot_sq - U_tot**2) / (n_s*(n_s - 1)))
        # Making sure the std has non NaNs
        U_test_std = np.nan_to_num(U_test_std)

        dirname = os.path.join(self.eqnPath, "data")
        print(f"Saving data to {dirname}")
        np.save(os.path.join(dirname, X_FILE), X)
        if self.has_t:
            np.save(os.path.join(dirname, T_FILE), t)
        np.save(os.path.join(dirname, U_MEAN_FILE), U_test_mean)
        np.save(os.path.join(dirname, U_STD_FILE), U_test_std)

        # Store the HiFi hyperparams
        hp_hifi = {}
        hp_hifi["n_x"] = self.n_x
        hp_hifi["x_min"] = x_min
        hp_hifi["x_max"] = x_max
        hp_hifi["n_t"] = self.n_t
        if self.has_y:
            hp_hifi["n_y"] = self.n_y
            hp_hifi["y_min"] = y_min
            hp_hifi["y_max"] = y_max
        if self.has_z:
            hp_hifi["n_z"] = self.n_z
            hp_hifi["z_min"] = z_min
            hp_hifi["z_max"] = z_max
        if self.has_t:
            hp_hifi["t_min"] = t_min
            hp_hifi["t_max"] = t_max
        hp_hifi["mu_min"] = mu_min.tolist()
        hp_hifi["mu_max"] = mu_max.tolist()
        hp_hifi["n_s"] = n_s
        with open(os.path.join(dirname, HP_FILE), "w") as f:
            json.dump(hp_hifi, f)
