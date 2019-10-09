import sys
import os
import numpy as np

eqnPath = "2d-ackley"
sys.path.append(eqnPath)
sys.path.append("utils")
import sys
import os
import numpy as np

eqnPath = "2d-ackley"
sys.path.append(eqnPath)
from shekelutils import plot_results, prep_data

from names import X_FILE, U_MEAN_FILE, U_STD_FILE

n_x = 400
n_y = 400
n_t = 1e6

U_h, _, _, _ = prep_data(n_x, n_y, int(n_t))

x = np.linspace(0, 10, n_e)
u_mean = np.mean(U_h, axis=1)
u_std = np.std(U_h, axis=1)

dirname = os.path.join(eqnPath, "data")
print(f"Saving data to {dirname}")
np.save(os.path.join(dirname, X_FILE), x)
np.save(os.path.join(dirname, U_MEAN_FILE), u_mean)
np.save(os.path.join(dirname, U_STD_FILE), u_std)
