import sys
import json
import numpy as np
import tensorflow as tf

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
from dataprep import prep_data
from regression import create_model_and_train
from predictions import predict_and_assess
from plots import plot_results


# HYPER PARAMETERS
if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    from hyperparams import hp

# DATA PREPARATION
X_v_train, v_train, X_v_val, v_val, \
    lb, ub, V, U_val = prep_data(hp,
            use_cache=False, save_cache=True,
            fast_pod=True)
    
# NN-REGRESSION TRAINING
model = create_model_and_train(X_v_train, v_train,
                               X_v_val, v_val,
                               hp, lb, ub, V, U_val)

# PREDICTIONS AND PERFORMANCE
U_val_struct, U_pred_struct = predict_and_assess(model,
                                                 X_v_val,
                                                 U_val, V, hp)

# PLOTTING AND SAVING RESULTS
plot_results(U_val_struct, U_pred_struct, hp, eqnPath)
plot_results(U_val_struct, U_pred_struct, hp)
