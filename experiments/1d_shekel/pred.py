"""POD-NN modeling for 1D Shekel Equation."""
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

#%% Predict and restruct
U_pred, U_pred_sig = model.predict(X_v_val)

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
n_samples = 3
mu_lhs_in = sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))

mu_lhs_out_min = sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
mu_lhs_out_max = sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
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
        U_pred_i, U_pred_i_sig = model.predict(X_i)
        ax = fig.add_subplot(gs[row, col])
        ax.plot(x, U_pred_i, "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
        ax.plot(x, U_samples[:, idx_i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
        lower = U_pred_i[:, 0] - 2*U_pred_i_sig[:, 0]
        upper = U_pred_i[:, 0] + 2*U_pred_i_sig[:, 0]
        ax.fill_between(x, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
        ax.set_xlabel("$x$")
        if col == 2:
            ax.legend()
# plt.show()
savefig("results/podensnn-shekel-graph-samples")
