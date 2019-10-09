import sys
import os
import numpy as np

eqnPath = "2d-ackley"
sys.path.append(eqnPath)
sys.path.append("utils")
from ackleyutils import plot_results, prep_data

from names import X_FILE, Y_FILE, U_MEAN_FILE, U_STD_FILE

n_x = 400
n_y = 400
n_s = int(1e2)
x_min = -5.
x_max = 5.
y_min = -5.
y_max = 5.

X, Y, U_h, _, _, _ = prep_data(n_x, n_y, n_s, x_min, x_max, y_min, y_max)

U_h_struct = np.reshape(U_h, (n_x, n_y, n_s))

u_mean = np.mean(U_h_struct, axis=2)
u_std = np.std(U_h_struct, axis=2)
print(u_mean.shape)

dirname = os.path.join(eqnPath, "data")
print(f"Saving data to {dirname}")
np.save(os.path.join(dirname, X_FILE), X)
np.save(os.path.join(dirname, Y_FILE), Y)
np.save(os.path.join(dirname, U_MEAN_FILE), u_mean)
np.save(os.path.join(dirname, U_STD_FILE), u_std)
