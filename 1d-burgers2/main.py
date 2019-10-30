import os
import sys
import json
import numpy as np
import tensorflow as tf

eqnPath = "1d-burgers2"
sys.path.append(eqnPath)
from dataprep import prep_data
from regression import create_model_and_train
from predictions import predict_and_assess
from plots import plot_results

sys.path.append(os.path.join(eqnPath, "utils"))
from podnn import PodnnModel
from metrics import error_podnn

# HYPER PARAMETERS
if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    from hyperparams import hp


class Burgers2PodnnModel(PodnnModel):
    # def u_1(x, mu):
    #     return x / (1 + np.exp(1/(4*mu)*(x**2 - 1/4)))

    def u(self, X, t, mu):
        x = X[0]
        t0 = np.exp(1 / (8*mu))
        return (x/t) / (1 + np.sqrt(t/t0)*np.exp(x**2/(4*mu*t)))


model = Burgers2PodnnModel(hp["n_v"], hp["n_x"], hp["n_t"], eqnPath)

X_v_train, v_train, \
    X_v_val, v_val, \
    U_val = model.generate_dataset(hp["x_min"], hp["x_max"],
                                   hp["t_min"], hp["t_max"],
                                   hp["mu_min"], hp["mu_max"],
                                   hp["n_s"],
                                   hp["train_val_ratio"],
                                   hp["eps"])

def error_val():
    return 0.0
    U_pred = model.predict(X_v_val)
    return error_podnn(U_val, U_pred)
model.train(X_v_train, v_train, error_val, hp["h_layers"],
            hp["epochs"], hp["lr"], hp["lambda"]) 

U_pred = model.predict(X_v_val)

U_pred_struct = model.restruct(U_pred)
U_val_struct = model.restruct(U_val)
 
# PLOTTING AND SAVING RESULTS
plot_results(U_val_struct, U_pred_struct, hp, eqnPath)
plot_results(U_val_struct, U_pred_struct, hp)
exit(0)


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
