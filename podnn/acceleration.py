"""Compiled and parallelized functions."""

import warnings
import numpy as np
from numba import jit, prange


# Disable bad division warning when summing up squares
warnings.filterwarnings("ignore", category=RuntimeWarning)


@jit(nopython=True, parallel=True)
def loop_vdot(n_s, U_tot, U_tot_sq, V, v_pred_hifi):
    """Return mean, std from parallelized dot product between V an v"""
    # pylint: disable=not-an-iterable
    for i in prange(n_s):
        # Computing one snapshot
        U = V.dot(v_pred_hifi[i])
        # Building the sum and the sum of squaes
        U_tot += U
        U_tot_sq += U**2
    return U_tot, U_tot_sq


@jit(nopython=True, parallel=True)
def loop_vdot_t(n_s, n_t, U_tot, U_tot_sq, V, v_pred_hifi):
    """Return mean, std from parallelized dot product between V an v (w/ t)."""
    # pylint: disable=not-an-iterable
    for i in prange(n_s):
        # Computing one snapshot
        s = n_t * i
        e = n_t * (i + 1)
        U = V.dot(v_pred_hifi[s:e].T)
        # Building the sum and the sum of squaes
        U_tot += U
        U_tot_sq += U**2
    return U_tot, U_tot_sq


@jit(nopython=True, parallel=True)
def loop_u(u, n_s, n_h, X_v, U, X, mu_lhs):
    """Return the inputs/snapshots matrices from parallel computation."""
    # pylint: disable=not-an-iterable
    for i in prange(n_s):
        X_v[i, :] = mu_lhs[i]
        U[:, i] = u(X, 0, mu_lhs[i, :]).reshape((n_h,))
    U_struct = U
    return X_v, U, U_struct


@jit(nopython=True, parallel=True)
def loop_u_t(u, n_s, n_t, n_v, n_xyz, n_h,
             X_v, U, U_struct, X, mu_lhs, t_min, t_max):
    """Return the inputs/snapshots matrices from parallel computation (w/ t)."""
    # Creating the time steps
    t = np.linspace(t_min, t_max, n_t)
    tT = t.reshape((n_t, 1))
    # pylint: disable=not-an-iterable
    for i in prange(n_s):
        # Getting the snapshot times indices
        s = n_t * i
        e = n_t * (i + 1)

        # Setting the regression inputs (t, mu)
        X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))

        # Calling the analytical solution function
        Ui = np.zeros((n_v, n_xyz, n_t))
        for j in range(n_t):
            Ui[:, :, :, j] = u(X, t[j], mu_lhs[i])

        U[:, s:e] = Ui.reshape((n_h, n_t))
        U_struct[:, :, i] = U[:, s:e]
    return X_v, U, U_struct


@jit(nopython=True, parallel=True)
def lhs(n, samples):
    """Borrowed, parallelized __lhscentered() from pyDOE."""

    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros(u.shape)
    for j in prange(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a

    # Make the random pairings
    H = np.zeros(rdpoints.shape)
    for j in prange(n):
        order = np.random.permutation(np.arange(samples))
        H[:, j] = rdpoints[order, j]

    return H
