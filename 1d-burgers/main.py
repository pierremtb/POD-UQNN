import sys
import json
import numpy as np
import tensorflow as tf

np.random.seed(1111)
tf.random.set_seed(1111)

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
sys.path.append("utils")
from dataprep import prep_data
from training import create_model_and_train
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
    lb, ub, V, U_val = prep_data(
        hp["n_x"], hp["x_min"], hp["x_max"],
        hp["n_t"], hp["t_min"], hp["t_max"],
        hp["n_s"], hp["mu_mean"],
        hp["train_val_ratio"], hp["eps"],
        use_cache=True)
    
# NN-REGRESSION TRAINING
model = create_model_and_train(X_v_train, v_train,
                               X_v_val, v_val,
                               lb, ub, V, U_val)

# PREDICTIONS AND PERFORMANCE
U_val_struct, U_pred_struct = predict_and_assess(model,
                                                 X_v_val,
                                                 U_val, V)

# PLOTTING AND SAVING RESULTS
plot_results(U_val_struct, U_pred_struct, hp, eqnPath)
plot_results(U_val_struct, U_pred_struct, hp)
