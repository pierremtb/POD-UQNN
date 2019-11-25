"""Module defining the TestGenerator class."""

import sys
import time
import os
import yaml
from tqdm.auto import tqdm
import numpy as np
import numba as nb
from numba import objmode, jit, prange

from .acceleration import lhs
from .mesh import create_linear_mesh

X_FILE = "X.npy"
T_FILE = "t.npy"
U_MEAN_FILE = "u_mean.npy"
U_STD_FILE = "u_std.npy"
HP_FILE = "HP.txt"


class TestGenerator:
    def __init__(self, u, n_v, n_x, n_y=0, n_z=0, n_t=0):
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
        """Equation-specific plot function, to be implemented to one's needs."""
        raise NotImplementedError

    def get_x_tuple(self):
        """Get the shape of the space domain."""
        tup = (self.n_x,)
        if self.has_y:
            tup += (self.n_y,)
            if self.has_z:
                tup += (self.n_z,)
        return tup

    def get_u_tuple(self):
        """Get the shape of the solution domain, includes space (and time)."""
        tup = self.get_x_tuple()
        if self.has_t:
            tup += (self.n_t,)
        return (self.n_v,) + tup

    def computeParallel(self, n_s, U_tot, U_tot_sq, X, t, mu_lhs):
        n_t = self.n_t
        u = nb.njit(self.u)

        pbar = tqdm(total=n_s)
        def bumpBar():
            pbar.update(1)

        @jit(nopython=True, parallel=True)
        def loop_t(n_s, n_t, U_tot, U_tot_sq, X, t, mu_lhs):
            for i in prange(n_s):
                # Computing one snapshot
                U = np.zeros_like(U_tot)
                for j in prange(n_t):
                    u_j = u(X, t[j], mu_lhs[i, :])
                    U[:, j] = u_j
                # Building the sum and the sum of squaes
                U_tot += U
                U_tot_sq += U**2
                with objmode():
                    bumpBar()
            return U_tot, U_tot_sq
        
        @jit(nopython=True, parallel=True)
        def loop(n_s, U_tot, U_tot_sq, X, mu_lhs):
            for i in prange(n_s):
                # Computing one snapshot
                U = u(X, 0, mu_lhs[i, :])
                # Building the sum and the sum of squaes
                U_tot += U
                U_tot_sq += U**2
                with objmode():
                    bumpBar()
            return U_tot, U_tot_sq
        
        if self.has_t:
            U_tot, U_tot_sq = loop_t(n_s, n_t, U_tot, U_tot_sq, X, t, mu_lhs)
        else: 
            U_tot, U_tot_sq = loop(n_s, U_tot, U_tot_sq, X, mu_lhs)

        with objmode():
            pbar.close()
        
        return U_tot, U_tot_sq

    def compute(self, n_s, U_tot, U_tot_sq, X, t, mu_lhs):
        for i in tqdm(range(n_s)):
            # Computing one snapshot
            U = np.zeros_like(U_tot)
            if self.has_t:
                for j in range(self.n_t):
                    U[:, j] = self.u(X, t[j], mu_lhs[i, :])
            else:
                U = self.u(X, 0, mu_lhs[i, :])

            # Building the sum and the sum of squaes
            U_tot += U
            U_tot_sq += U**2

        return U_tot, U_tot_sq

    def generate(self, n_s, mu_min, mu_max, x_min, x_max,
                y_min=0, y_max=0, z_min=0, z_max=0,
                t_min=0, t_max=0, parallel=False):
        """Generate a hifi-test solution of the problem's equation."""
        mu_min, mu_max = np.array(mu_min), np.array(mu_max)

        # Static data
        x_mesh = create_linear_mesh(x_min, x_max, self.n_x,
                                    y_min, y_max, self.n_y,
                                    z_min, z_max, self.n_z)

        # Getting the nodes coordinates
        X = x_mesh[:, 1:].T
        n_xyz = x_mesh.shape[0]

        # Generating time steps
        t = None
        if self.has_t:
            t = np.linspace(t_min, t_max, self.n_t)

        # Number of DOFs and of non-spatial params
        n_h = self.n_v * n_xyz
        n_p = mu_min.shape[0]

        # Number of inputs (time + number of non-spatial params)
        n_d = n_p
        if self.has_t:
            n_d += 1

        # Lower and upper bound
        lb = mu_min
        ub = mu_max

        # The sum and sum of squares recipient vectors
        if self.has_t:
            U_tot = np.zeros((n_h, self.n_t))
            U_tot_sq = np.zeros((n_h, self.n_t))
        else:
            U_tot = np.zeros((n_h,))
            U_tot_sq = np.zeros((n_h,))

        # Parameters sampling
        X_mu = lhs(n_s, n_p).T
        mu_lhs = lb + (ub - lb)*X_mu

        # Going through the snapshots one by one without saving them
        if parallel:
            U_tot, U_tot_sq = self.computeParallel(n_s, U_tot, U_tot_sq, X, t, mu_lhs)
        else:
            U_tot, U_tot_sq = self.compute(n_s, U_tot, U_tot_sq, X, t, mu_lhs)
        
        # Recreating the mean and the std
        U_test_mean = U_tot / n_s
        U_test_std = np.sqrt((n_s*U_tot_sq - U_tot**2) / (n_s*(n_s - 1)))
        # Making sure the std has non NaNs
        U_test_std = np.nan_to_num(U_test_std)

        # Reshaping
        X_out = []
        X_out.append(X[0].reshape(self.get_x_tuple()))
        if self.has_y:
            X_out.append(X[1].reshape(self.get_x_tuple()))
            if self.has_z:
                X_out.append(X[2].reshape(self.get_x_tuple()))
        U_test_mean = np.reshape(U_test_mean, self.get_u_tuple())
        U_test_std = np.reshape(U_test_std, self.get_u_tuple())

        dirname = "data" 
        print(f"Saving data to {dirname}")
        np.save(os.path.join(dirname, X_FILE), X_out)
        if self.has_t:
            np.save(os.path.join(dirname, T_FILE), t)
        np.save(os.path.join(dirname, U_MEAN_FILE), U_test_mean)
        np.save(os.path.join(dirname, U_STD_FILE), U_test_std)

        # Store the HiFi hyperparams
        HP_hifi = {}
        HP_hifi["n_x"] = self.n_x
        HP_hifi["x_min"] = x_min
        HP_hifi["x_max"] = x_max
        HP_hifi["n_t"] = self.n_t
        if self.has_y:
            HP_hifi["n_y"] = self.n_y
            HP_hifi["y_min"] = y_min
            HP_hifi["y_max"] = y_max
        if self.has_z:
            HP_hifi["n_z"] = self.n_z
            HP_hifi["z_min"] = z_min
            HP_hifi["z_max"] = z_max
        if self.has_t:
            HP_hifi["t_min"] = t_min
            HP_hifi["t_max"] = t_max
        HP_hifi["mu_min"] = mu_min.tolist()
        HP_hifi["mu_max"] = mu_max.tolist()
        HP_hifi["n_s"] = n_s
        with open(os.path.join(dirname, HP_FILE), "w") as f:
             yaml.dump(HP_hifi, f)
