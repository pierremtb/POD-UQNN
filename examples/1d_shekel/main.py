"""POD-NN modeling for 1D Shekel Equation."""
#%% Import
%reload_ext autoreload
%autoreload 2
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.plotting import genresultdir

sys.path.append(".")
from genhifi import u, generate_test_dataset
from plot import plot_results

#%% Prepare
from hyperparams import HP as hp
resdir = genresultdir()

# generate_test_dataset()

x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, _, _, \
    X_v_test, v_test, U_test = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                                hp["n_s"],
                                                hp["train_val_test"],
                                                eps=hp["eps"], n_L=hp["n_L"],
                                                u_noise=hp["u_noise"],
                                                x_noise=hp["x_noise"])

#%% Train
model.initBNN(hp["h_layers"],
                hp["lr"], 1/X_v_train.shape[0],
                hp["norm"])
#%%
train_res = model.train(X_v_train, v_train, hp["epochs"],
                        hp["train_val_test"], freq=hp["log_frequency"], silent=True)
#%% Predict and restruct
U_pred = model.predict(X_v_test)
U_pred = model.restruct(U_pred)
U_test = model.restruct(U_test)

x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
# lower = U_pred - 3 * U_pred_sig
# upper = U_pred + 3 * U_pred_sig
# plt.fill_between(x, lower[:, 0], upper[:, 0], 
#                     facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
import matplotlib.pyplot as plt
plt.plot(x, U_pred[0, :, 0], "b-")
plt.plot(x, U_test[0, :, 0], "r--")
# plt.plot(x, model.predict(X_v_test)[:, 0])
plt.show()

#%% Hifi predictions
# Sample the new model to generate a HiFi prediction
print("Sampling {n_s_hifi} parameters")
X_v_test_hifi = model.generate_hifi_inputs(hp["n_s_hifi"],
                                            hp["mu_min"], hp["mu_max"])
print("Predicting the {n_s_hifi} corresponding solutions")
U_pred_hifi, U_pred_hifi_sig = model.predict_var(X_v_test_hifi)
U_pred_hifi_mean = (model.restruct(U_pred_hifi.mean(-1), no_s=True),
                    model.restruct(U_pred_hifi_sig.mean(-1), no_s=True))
U_pred_hifi_std = (model.restruct(U_pred_hifi.std(-1), no_s=True),
                    model.restruct(U_pred_hifi_sig.mean(-1), no_s=True))
sigma_pod = model.pod_sig.mean()

# Plot against test and save
plot_results(U_test, U_pred, U_pred_hifi_mean, U_pred_hifi_std, sigma_pod,
             resdir, train_res, hp)
