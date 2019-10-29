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

# HYPER PARAMETERS
if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    from hyperparams import hp


class Burgers2PodnnModel(PodnnModel):
    # def u_1(x, mu):
    #     return x / (1 + np.exp(1/(4*mu)*(x**2 - 1/4)))


    def u(x, t, mu):
        t0 = np.exp(1 / (8*mu))
        return (x/t) / (1 + np.sqrt(t/t0)*np.exp(x**2/(4*mu*t)))
    
model = Burgers2PodnnModel(hp["n_v"], hp["n_x"], hp["n_t"]) 
X_v_train, v_train, \
        X_v_val, v_val, U_val = model.generate_dataset(hp["x_min"], hp["x_max"],
                                                       hp["t_min"], hp["t_max"],
                                                       hp["mu_min"], hp["mu_max"],
                                                       hp["n_s"], hp["train_val_ratio"],
                                                       hp["eps"])
model.train(X_v_train, v_train, error_val, hp["h_layers"],
            hp["tf_epochs"], hp["learning_rate"], hp["lambda"]) 
 
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
