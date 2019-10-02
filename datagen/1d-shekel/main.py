import sys
import os
import numpy as np

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
from pod import prep_data
from shekelutils import plot_results

U_h, _, _, _ = prep_data(n_e=300,
                         n_t=int(1e6),
                         bet_count=10,
                         gam_count=10)

np.save(os.path.join(eqnPath, "data", "U_h.npy"), U_h) 
plot_results(U_h)

