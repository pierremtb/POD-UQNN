"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.plotting import figsize, savefig
from poduqnn.handling import sample_mu

from hyperparams import HP as hp
from hyperparams import u

#%% Load models
model = PodnnModel.load("cache")
# X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

# #%% Predict and restruct
# U_pred, U_pred_sig = model.predict(X_v_val)

# #%% Validation metrics
# U_pred, _ = model.predict(X_v_val)
# err_val = re_s(U_val, U_pred)
# print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
# mu_lhs = sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# X_v_tst, U_tst, _, _ = \
#     model.create_snapshots(model.n_d, model.n_h, u, mu_lhs,
#                            t_min=hp["t_min"], t_max=hp["t_max"])
# U_pred, U_pred_sig = model.predict(X_v_tst)
# print(f"RE_tst: {re_s(U_tst, U_pred):4f}")
# U_tst = model.restruct(U_tst)[0]
# U_pred = model.restruct(U_pred)[0]

#%% Samples graph
hp["mu_min_out"] = [0.0005]
hp["mu_max_out"] = [0.0105]
n_samples = 3
# mu_lhs_in = sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# mu_lhs_out_min = sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
# mu_lhs_out_max = sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
# mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))
# np.save(os.path.join("cache", "mu_lhs_in.npy"), mu_lhs_in)
# np.save(os.path.join("cache", "mu_lhs_out.npy"), mu_lhs_out)
mu_lhs_in = np.load(os.path.join("cache", "mu_lhs_in.npy"))
mu_lhs_out = np.load(os.path.join("cache", "mu_lhs_out.npy"))
times = [25, 75]

#%% Contours for demo
n_plot_x = 2
n_plot_y = 3
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
xxT, ttT = np.meshgrid(x, t)
xx, tt = xxT.T, ttT.T
XT = np.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))

X_v_samples, _, U_samples, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs_in,
                           t_min=hp["t_min"], t_max=hp["t_max"])
X_v_samples_out, _, U_samples_out, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs_out,
                           t_min=hp["t_min"], t_max=hp["t_max"])

U_pred, U_pred_sig = model.predict(X_v_samples)
U_pred_out, U_pred_sig_out = model.predict(X_v_samples_out)
U_pred, U_pred_sig = model.restruct(U_pred), model.restruct(U_pred_sig)
U_pred_out, U_pred_sig_out = model.restruct(U_pred_out), model.restruct(U_pred_sig_out)
# idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
idx = 0

# First column: contour plot of the true value
ax = fig.add_subplot(gs[0, 0])
U_tst_grid = griddata(XT, U_samples[..., idx].flatten(), (xx, tt), method='cubic')
print(U_tst_grid.shape)
h = ax.imshow(U_tst_grid, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.axvline(X_v_samples[times[0], 0], color="w", ls="-.")
ax.set_title(r"$u_D(s=" + f"{X_v_samples[0, 1]:.4f}" + ")$")

# First column: contour plot of the predicted mean
ax = fig.add_subplot(gs[1, 0])
U_pred_grid = griddata(XT, U_pred[..., idx].flatten(), (xx, tt), method='cubic')
h = ax.imshow(U_pred_grid, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.axvline(X_v_samples[times[1], 0], color="w", ls="-.")
ax.set_title(r"$\hat{u}^\mu_D(s=" + f"{X_v_samples[0, 1]:.4f}" + ")$")

# Slices
def plot_slice(row, col, t, lbl, X_v, U_pred_i, U_pred_i_sig, U_true_i):
    ax = fig.add_subplot(gs[row, col + 1])
    ax.plot(x, U_pred_i[:, t], "b-", label=r"$\hat{u}^\mu_D(s_{" + lbl + r"})$")
    ax.plot(x, U_true_i[:, t], "r--", label=r"$u_D(s_{" + lbl + r"})$")
    lower = U_pred_i[:, t] - 2*U_pred_i_sig[:, t]
    upper = U_pred_i[:, t] + 2*U_pred_i_sig[:, t]
    ax.fill_between(x, lower, upper, alpha=0.2, label=r"$\pm2\hat{u}^\sigma_D(s_{" + lbl + r"})$")
    ax.set_xlabel(f"$x$")
    if row == 0:
        ax.legend()
    st = hp["n_t"] * col
    en = hp["n_t"] * (col + 1)
    X_i = X_v[st:en, :]

    if row == 0:
        ax.set_title(r"$s=" + f"{X_i[0, 1]:.4f}" + r" \in \Omega,\ "
                        + f"t={X_i[time, 0]:.2f}$")
    else:
        ax.set_title(r"$s=" + f"{X_i[0, 1]:.4f}" + r" \in \Omega{\footnotesize\textrm{out}},\ "
                        + f"t={X_i[time, 0]:.2f}$")

for row, time in enumerate(times):
    lbl = r"{\scriptscriptstyle\textrm{tst}}" if col == 0 else r"{\scriptscriptstyle\textrm{out}}"
    X_v = X_v_samples
    U_pred_i, U_pred_i_sig = U_pred[0, :, :, idx], U_pred_sig[0, :, :, idx]
    U_true_i = U_samples[:, :, idx]
    plot_slice(row, 0, time, lbl, X_v, U_pred_i, U_pred_i_sig, U_true_i)
    X_v = X_v_samples_out
    U_pred_i, U_pred_i_sig = U_pred_out[0, :, :, idx], U_pred_sig_out[0, :, :, idx]
    U_true_i = U_samples_out[:, :, idx]
    plot_slice(row, 1, time, lbl, X_v, U_pred_i, U_pred_i_sig, U_true_i)
plt.tight_layout()
savefig("results/podbnn-burger-graph-meansamples")
