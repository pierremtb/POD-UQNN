"""POD-NN modeling for 2D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_s

from hyperparams import HP as hp
from hyperparams import u
from podnn.plotting import figsize, savefig

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

#%% Predict and restruct
U_pred, U_pred_sig = model.predict(X_v_val)

#%% Validation metrics
U_pred, _ = model.predict(X_v_val)
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
mu_lhs = model.sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
X_v_tst, U_tst, _, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
U_pred, U_pred_sig = model.predict(X_v_tst)
print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

#%% Samples graph
n_samples = 2
mu_lhs_in = model.sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
mu_lhs_out_min = model.sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
mu_lhs_out_max = model.sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))


# Contours for demo
n_plot_x = 2
n_plot_y = 3
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
y = np.linspace(hp["y_min"], hp["y_max"], hp["n_y"])
xxT, yyT = np.meshgrid(x, y)
xx, yy = xxT.T, yyT.T
ax = fig.add_subplot(gs[0, 0])
U_tst = np.reshape(U_tst, (hp["n_x"], hp["n_y"], -1))
levels = list(range(2, 15))
ct = ax.contourf(xx, yy, U_tst.mean(-1), levels=levels, origin="lower")
plt.colorbar(ct)
ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
ax.axis("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax = fig.add_subplot(gs[1, 0])
U_pred = np.reshape(U_pred, (hp["n_x"], hp["n_y"], -1))
levels = list(range(2, 15))
ct = ax.contourf(xx, yy, U_pred.mean(-1), levels=levels, origin="lower")
plt.colorbar(ct)
ax.axis("equal")
ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
# plt.show()
# savefig("results/podensnn-ackley-graph-means")

# Slices
# n_plot_x = 2
# n_plot_y = n_samples
# fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
# gs = fig.add_gridspec(n_plot_x, n_plot_y)
for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
    X_v_samples, U_samples, _, _ = \
        model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
    U_samples = np.reshape(U_samples, (hp["n_x"], hp["n_y"], -1))
                            
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
    for col, idx_i in enumerate(idx):
        lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
        X_i = X_v_samples[idx_i, :].reshape(1, -1)
        U_pred_i, U_pred_i_sig = model.predict(X_i)
        U_pred_i = np.reshape(U_pred_i, (hp["n_x"], hp["n_y"], -1))
        U_pred_i_sig = np.reshape(U_pred_i_sig, (hp["n_x"], hp["n_y"], -1))
        ax = fig.add_subplot(gs[row, col+1])
        ax.plot(x, U_pred_i[:, 199, 0], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
        ax.plot(x, U_samples[:, 199, idx_i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
        lower = U_pred_i[:, 199, 0] - 2*U_pred_i_sig[:, 199, 0]
        upper = U_pred_i[:, 199, 0] + 2*U_pred_i_sig[:, 199, 0]
        ax.fill_between(x, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
        ax.set_xlabel("$x\ (y=0)$")
        if col == len(idx) - 1:
            ax.legend()
plt.tight_layout()
# plt.show()
savefig("results/podensnn-ackley-graph-meansamples")
