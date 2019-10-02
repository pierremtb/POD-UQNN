import sys
import numpy as np

eqnPath = "1d-shekel"
sys.path.append(eqnPath)
from pod import prep_data
from shekelutils import plot_results

U_h, _, _, _ = prep_data(n_e=300,
                         n_t=int(1e6),
                         bet_count=0,
                         gam_count=3)

plot_results(U_h)

