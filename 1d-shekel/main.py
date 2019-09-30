import sys
import numpy as np

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
from pod import get_pod_bases, prep_data

# Space params
dx = 1/30
n_e = int(10/dx)

# Snapshots
n_t = 100

# POD stopping param
eps = 1e-10

# Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
# u_rb are the reduced coefficients we're looking for
x, S = prep_data(n_e, n_t)
V = get_pod_bases(S, n_e, n_t, eps)

# Projecting
u_h_train = V.dot(V.T).dot(S)
