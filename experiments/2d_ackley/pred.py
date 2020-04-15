"""POD-NN modeling for 2D Ackley Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.handling import sample_mu
from poduqnn.plotting import figsize, savefig

from hyperparams import HP as hp
from hyperparams import u

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

#%% Validation metrics
U_pred, _ = model.predict(X_v_val)
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
mu_lhs = sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
X_v_tst, U_tst, _, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
U_pred, U_pred_sig = model.predict(X_v_tst)
print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

#%% Samples graph
n_samples = 2
# mu_lhs_in = sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# mu_lhs_out_min = sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
# mu_lhs_out_max = sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
# mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))
# np.save(os.path.join("cache", "mu_lhs_in.npy"), mu_lhs_in)
# np.save(os.path.join("cache", "mu_lhs_out.npy"), mu_lhs_out)
mu_lhs_in = np.load(os.path.join("cache", "mu_lhs_in.npy"))
mu_lhs_out = np.load(os.path.join("cache", "mu_lhs_out.npy"))


# Contours for demo
n_plot_x = 2
n_plot_y = 3
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
y = np.linspace(hp["y_min"], hp["y_max"], hp["n_y"])
xxT, yyT = np.meshgrid(x, y)
xx, yy = xxT.T, yyT.T

X_v_samples, _, U_samples, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs_in)
X_v_samples_out, _, U_samples_out, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs_out)

U_samples = np.reshape(U_samples, (hp["n_x"], hp["n_y"], -1))
U_samples_out = np.reshape(U_samples_out, (hp["n_x"], hp["n_y"], -1))
U_pred, U_pred_sig = model.predict(X_v_samples)
U_pred_out, U_pred_sig_out = model.predict(X_v_samples_out)
U_pred, U_pred_sig = model.restruct(U_pred), model.restruct(U_pred_sig)
U_pred = np.reshape(U_pred, (hp["n_x"], hp["n_y"], -1))
U_pred_sig = np.reshape(U_pred_sig, (hp["n_x"], hp["n_y"], -1))
U_pred_out, U_pred_sig_out = model.restruct(U_pred_out), model.restruct(U_pred_sig_out)
U_pred_out = np.reshape(U_pred_out, (hp["n_x"], hp["n_y"], -1))
U_pred_sig_out = np.reshape(U_pred_sig_out, (hp["n_x"], hp["n_y"], -1))
# idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
idx = [0, 1]

# First column: contour plot of the true value
ax = fig.add_subplot(gs[0, 0])
levels = list(range(2, 15))
ct = ax.contourf(xx, yy, U_samples[..., idx[0]], levels=levels, origin="lower")
plt.colorbar(ct)
ax.set_title(r"$u{\scriptsize\textrm{D}}([" + f"{X_v_samples[0, 0]:.2f}," + f"{X_v_samples[0, 1]:.2f}," + f"{X_v_samples[0, 2]:.2f}] )$")
ax.axis("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axhline(0., color="w", ls="-.")
ax = fig.add_subplot(gs[1, 0])
levels = list(range(2, 15))
ct = ax.contourf(xx, yy, U_pred[..., idx[0]], levels=levels, origin="lower")
plt.colorbar(ct)
ax.axis("equal")
ax.set_title(r"$\hat{u}^\mu_D([" + f"{X_v_samples[0, 0]:.2f}," + f"{X_v_samples[0, 1]:.2f}," + f"{X_v_samples[0, 2]:.2f}] )$")
ax.set_xlabel("$x$")
ax.axhline(0., color="w", ls="-.")
ax.set_ylabel("$y$")

# Slices
def plot_slice(row, col, lbl, X_v, U_pred_i, U_pred_i_sig, U_true_i):
    print(U_pred_i.shape)
    ax = fig.add_subplot(gs[row, col + 1])
    ax.plot(x, U_pred_i[:, 199], "b-", label=r"$\hat{u}^\mu_D(s_{" + lbl + r"})$")
    ax.plot(x, U_true_i[:, 199], "r--", label=r"$u_D(s_{" + lbl + r"})$")
    lower = U_pred_i[:, 199] - 2*U_pred_i_sig[:, 199]
    upper = U_pred_i[:, 199] + 2*U_pred_i_sig[:, 199]
    ax.fill_between(x, lower, upper, alpha=0.2, label=r"$\pm 2\hat{u}^\sigma_D(s_{" + lbl + r"})$")
    ax.set_xlabel("$x\ (y=0)$")
    title_st = r"$s=[" + f"{X_v[0, 0]:.2f}," + f"{X_v[0, 1]:.2f}," + f"{X_v[0, 2]:.2f}] "
    title_st += r"\in \Omega{\footnotesize\textrm{out}}$" if col + 1 == 2 else r"\in \Omega$"
    ax.set_title(title_st)
    if row == 0:
        ax.legend()

for row, idx_i in enumerate(idx):
    col = 0
    lbl = r"{\scriptscriptstyle\textrm{tst}}"
    X_v = X_v_samples
    U_pred_i, U_pred_i_sig = U_pred[..., idx_i], U_pred_sig[..., idx_i]
    U_true_i = U_samples[..., idx_i]
    plot_slice(row, col, lbl, X_v, U_pred_i, U_pred_i_sig, U_true_i)
    col = 1
    lbl = r"{\scriptscriptstyle\textrm{out}}"
    X_v = X_v_samples_out
    U_pred_i, U_pred_i_sig = U_pred_out[..., idx_i], U_pred_sig_out[..., idx_i]
    U_true_i = U_samples_out[..., idx_i]
    plot_slice(row, col, lbl, X_v, U_pred_i, U_pred_i_sig, U_true_i)

plt.tight_layout()
plt.show()
# savefig(os.path.join("results", "podbnn-ackley-graph-meansamples"))
