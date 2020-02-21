"""POD-NN modeling for 1D Shekel Equation."""
#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.metrics import re_s
from podnn.plotting import savefig, figsize

#%% Prepare
from hyperparams import HP as hp
from hyperparams import u
print(hp)

resdir = "cache"
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, _, \
    X_v_val, v_val, U_val = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                                hp["n_s"],
                                                hp["train_val"],
                                                eps=hp["eps"], n_L=hp["n_L"],
                                                u_noise=hp["u_noise"],
                                                x_noise=hp["x_noise"])

#%% Train
model.initBNN(hp["h_layers"],
                hp["lr"], 1/X_v_train.shape[0],
                hp["norm"])
train_res = model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
                        freq=hp["log_frequency"],
                        silent=False)

#%% Validation metrics
U_pred, _ = model.predict(X_v_val, samples=100)
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

import matplotlib.pyplot as plt
v_pred, v_pred_sig = model.predict_v(X_v_val)
print(v_pred)
# dist = model.regnn.model(X_v_val)
# v_pred, v_pred_sig = dist.mean().numpy(), np.sqrt(dist.variance())
print(v_pred.shape)
x = np.arange(v_pred.shape[1])
plt.plot(x, v_pred[0])
plt.plot(x, v_val[0])
plt.fill_between(x, v_pred[0] - 2*v_pred_sig[0],
                    v_pred[0] + 2*v_pred_sig[0], alpha=0.3)
plt.show()

# U_pred = model.restruct(U_pred)
# U_val = model.restruct(U_val)
x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
# lower = U_pred - 3 * U_pred_sig
# upper = U_pred + 3 * U_pred_sig
# plt.fill_between(x, lower[:, 0], upper[:, 0], 
#                     facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
plt.plot(x, U_pred[0], "b-")
plt.plot(x, U_val[0], "r--")
# plt.plot(x, model.predict(X_v_test)[:, 0])
plt.show()

#%% Sample the new model to generate a test prediction
mu_lhs = model.sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
X_v_tst, U_tst, _, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
U_pred, U_pred_sig = model.predict(X_v_tst, samples=100)
print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

#%% Samples graph
n_samples = 3
# mu_lhs_in = model.sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
idx = [[100, 250, 400], [50, 150, 250]]
mu_lhs_in = model.sample_mu(hp["n_s"], np.array(hp["mu_min"]), np.array(hp["mu_max"]), indices=idx[0])

# mu_lhs_out_min = model.sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
mu_lhs_out_min = model.sample_mu(hp["n_s_tst"], np.array(hp["mu_min_out"]), np.array(hp["mu_min"]), indices=idx[1])
# mu_lhs_out_max = model.sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
mu_lhs_out_max = model.sample_mu(hp["n_s_tst"], np.array(hp["mu_max"]), np.array(hp["mu_max_out"]), indices=idx[1])
mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))

n_plot_x = 2
n_plot_y = n_samples
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
    X_v_samples, U_samples, _, _ = \
        model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
                            
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    for col, idx_i in enumerate([0, 1, 2]):
        lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
        X_i = X_v_samples[idx_i, :].reshape(1, -1)
        U_pred_i, U_pred_i_sig = model.predict(X_i, samples=100)
        ax = fig.add_subplot(gs[row, col])
        ax.plot(x, U_pred_i, "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
        ax.plot(x, U_samples[:, idx_i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
        lower = U_pred_i[:, 0] - 2*U_pred_i_sig[:, 0]
        upper = U_pred_i[:, 0] + 2*U_pred_i_sig[:, 0]
        ax.fill_between(x, lower, upper, alpha=0.2, label=r"$3\sigma_D(s_{" + lbl + r"})$")
        ax.set_xlabel("$x$")
        if col == len(idx) - 1:
            ax.legend()
# plt.show()
savefig("results/graph-samples")
