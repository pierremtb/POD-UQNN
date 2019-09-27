import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
import os

start_time = time.time()

# Space params
dx = 1/30
n_e = int(10/dx)

# Shekel parameters (t=10-sized)
bet = 1/10 * np.array([[1, 2, 2, 4, 4, 6, 3, 7, 5, 5]]).T
gam = 1. * np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7]]).T

# Number of solutions
n_s = 100
print("n_s: ", n_s)

# Stopping parameter
eps = 1e-10
print("eps: ", eps)

# Perturbations
p_var = np.array([
    [4, 0.4],
    [1, 0.1],
    [8, 0.8],
])

# LHS sampling (first uniform, then perturbated)
X = lhs(n_s, p_var.shape[0]).T
lb = p_var[:, 0] - np.sqrt(3)*p_var[:, 1]
ub = p_var[:, 0] + np.sqrt(3)*p_var[:, 1]
R_var = lb + (ub - lb)*X

# Defining the u_h function (here Shekel)
def u_h(kxsi, gam, beta):
    S_i = np.zeros((n_e))
    gam[0:kxsi.shape[0], 0] = kxsi
    for j in range(n_e):
        S_i[j] = 0.
        for p in range(bet.shape[1]):
            S_i[j] -= 1/((x[j]-gam[0, p])**2 + bet[0, p])
    return S_i

# Creating the snapshots
S = np.zeros((n_e, n_s))
x = np.linspace(0, 10, n_e)
for i in range(n_s):
    S[:, i] = u_h(R_var[i, :], gam, bet)

# Performing SVD
W, D, ZT = np.linalg.svd(S, full_matrices=False)

# Getting MATLAB-like orientation
Z = ZT.T

# Storing eigenvalues and their sum
lambdas = D**2
sum_lambdas = np.sum(lambdas)

# Finding n_L
n_L = 0
sum_lambdas_trunc = 0.
for i in range(n_s):
    sum_lambdas_trunc += lambdas[i]
    n_L += 1
    if sum_lambdas_trunc/sum_lambdas >= (1 - eps):
        break

# Truncating according to n_L
lambdas_trunc = lambdas[0:n_L]

# Constructiong the reduced POD base V
V = np.zeros((n_e, n_L))
for i in range(n_L):
    V[:, i] = S.dot(Z[:, i]) / np.sqrt(lambdas_trunc[i])

print(f"Elapsed time is {time.time() - start_time} seconds.")
print(f"L = {n_L}")

# Saving
name = f"shek_Pod_bases_lhs_nxy_{n_e}_ns_{n_s}_epsi_{eps}.txt"
np.savetxt(os.path.join("1d-shekel", "results", name), V, delimiter="   ")
print(f"Written {name}")

def do_plots():    
    # Plotting snapshots
    for i in range(S.shape[1]):
        plt.plot(x, S[:, i])
    plt.plot(x, np.mean(S, axis=1))
    plt.plot(x, np.std(S, axis=1))
    plt.show()
    
    for i in range(n_L):
        plt.plot(x, V[:, i])
    plt.show()
#do_plots()
