import sys
import os
import numpy as np

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
from shekelutils import plot_results, prep_data

from names import X_FILE, U_MEAN_FILE, U_STD_FILE

n_x = 300
n_s = 1e6

U, _, _, _ = prep_data(n_x, int(n_s),
                         bet_count=10,
                         gam_count=10)

x = np.linspace(0, 10, n_x)
u_mean = np.mean(U, axis=1)
u_std = np.std(U, axis=1)

dirname = os.path.join(eqnPath, "data")
print(f"Saving data to {dirname}")
np.save(os.path.join(dirname, X_FILE), x)
np.save(os.path.join(dirname, U_MEAN_FILE), u_mean)
np.save(os.path.join(dirname, U_STD_FILE), u_std)
