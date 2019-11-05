import numpy as np
import sys
import os
from tqdm import tqdm
import pickle
import json
from pyDOE import lhs

sys.path.append("utils")
from mesh import create_linear_mesh

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

  def get_x_tuple(self):
    tup = (self.n_x,)
    if self.has_y:
      tup += (self.n_y,)
      if self.has_z:
        tup += (self.n_z,)
    return tup

  def get_u_tuple(self):
    tup = self.get_x_tuple()
    if self.has_t:
      tup += (self.n_t,)
    return (self.n_v,) + tup

  def generate(self, n_s, mu_min, mu_max, x_min, x_max,
              y_min=0, y_max=0, z_min=0, z_max=0,
              t_min=0, t_max=0):

    mu_min, mu_max = np.array(mu_min), np.array(mu_max)

    # Static data
    x_mesh = create_linear_mesh(x_min, x_max, self.n_x,
                                y_min, y_max, self.n_y,
                                z_min, z_max, self.n_z)

    # Getting the nodes coordinates
    X = x_mesh[:, 1:].T
    n_nodes = x_mesh.shape[0]

    # Generating time steps
    if self.has_t:
      t = np.linspace(t_min, t_max, self.n_t)

    # Number of DOFs and of non-spatial params
    n_h = self.n_v * n_nodes
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

    # Going through the snapshots one by one without saving them
    for i in tqdm(range(n_s)):
      # Computing one snapshot
      X_mu = lhs(n_s, n_p).T
      mu_lhs = lb + (ub - lb)*X_mu
      U = np.zeros_like(U_tot)
      if self.has_t:
        for j in range(self.n_t):
          U[:, j] = self.u(X, t[j], mu_lhs[i, :])
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

    # Reshaping
    X_out = []
    X_out.append(X[0].reshape(self.get_x_tuple()))
    if self.has_y:
      X_out.append(X[1].reshape(self.get_x_tuple()))
      if self.has_z:
        X_out.append(X[1].reshape(self.get_x_tuple()))
    U_test_mean = np.reshape(U_test_mean, self.get_u_tuple())
    U_test_std = np.reshape(U_test_std, self.get_u_tuple())

    dirname = os.path.join(self.eqnPath, "data")
    print(f"Saving data to {dirname}")
    np.save(os.path.join(dirname, X_FILE), X_out)
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
