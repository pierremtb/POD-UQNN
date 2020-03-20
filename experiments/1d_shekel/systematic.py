"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from lib.podnnmodel import PodnnModel
from lib.metrics import re_s
from lib.mesh import create_linear_mesh
from lib.handling import sample_mu

#%% Prepare
from hyperparams import HP as hp
from hyperparams import u

# Create linear space mesh
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))

# Study parameters
list_samples = [100, 300, 600, 1000]
list_epochs = [5000, 15000, 25000, 35000]

# Results containers
errors_val = np.zeros((len(list_samples), len(list_epochs)))
errors_tst = np.zeros((len(list_samples), len(list_epochs)))

for i, n_s in enumerate(list_samples):
    for j, epochs in enumerate(list_epochs):

        hp["epochs"] = epochs
        hp["n_s"] = n_s

        #%% Init the model
        model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])

        #%% Generate the dataset from the mesh and params
        X_v_train, v_train, \
            X_v_val, v_val, \
            U_val = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                           hp["n_s"],
                                           hp["train_val"],
                                           hp["eps"])

        #%% Train
        model.initNN(hp["h_layers"], hp["lr"], hp["lambda"])
        train_res = model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
                                hp["log_frequency"])

        #%% Validation metrics
        U_pred = model.predict(X_v_val)
        err_val = re_s(U_val, U_pred)
        print(f"RE_v: {err_val:4f}")
        errors_val[i, j] = err_val

        #%% Sample the new model to generate a test prediction
        mu_lhs = sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
        X_v_tst, U_tst, _ = \
            model.create_snapshots(mu_lhs.shape[0], mu_lhs.shape[0], model.n_d, model.n_h, u, mu_lhs)
        U_pred = model.predict(X_v_tst)
        err_tst = re_s(U_tst, U_pred)
        print(f"RE_tst: {err_tst:4f}")
        errors_tst[i, j] = err_tst

# Saving the results
np.savetxt(os.path.join("results", "systematic", "n_s.csv"),
        list_samples)
np.savetxt(os.path.join("results", "systematic", "tf_epochs.csv"),
        list_epochs)
np.savetxt(os.path.join("results", "systematic", "err_t_mean.csv"),
        errors_val)
np.savetxt(os.path.join("results", "systematic", "err_t_std.csv"),
        errors_tst)
