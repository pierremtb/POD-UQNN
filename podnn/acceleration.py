"""Compiled and parallelized functions."""

import warnings
import numpy as np
from numba import objmode, jit, prange


warnings.filterwarnings("ignore", category=RuntimeWarning) 

# TODO: implement pbar for accelerated loops
# pbar = tqdm(total=n_s, disable=True)
# def bumpBar():
#     pbar.update(1)

@jit(nopython=True, parallel=True)
def loop_vdot(n_s, U_tot, U_tot_sq, V, v_pred_hifi):
    for i in prange(n_s):
        # Computing one snapshot
        U = V.dot(v_pred_hifi[i])
        # Building the sum and the sum of squaes
        U_tot += U
        U_tot_sq += U**2
    return U_tot, U_tot_sq

@jit(nopython=True, parallel=True)
def loop_vdot_t(n_s, n_t, U_tot, U_tot_sq, V, v_pred_hifi):
    for i in prange(n_s):
        # Computing one snapshot
        s = n_t * i
        e = n_t * (i + 1)
        U = V.dot(v_pred_hifi[s:e].T)
        # Building the sum and the sum of squaes
        U_tot += U
        U_tot_sq += U**2
    return U_tot, U_tot_sq