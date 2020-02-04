"""POD-NN modeling for 2D Ackley Equation."""

#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_s
from podnn.mesh import create_linear_mesh
from podnn.plotting import figsize, savefig

#%% Prepare
from hyperparams import HP as hp

def u(X, _, mu):
    x = X[0]
    y = X[1]
    u_0 = - 20*(1+.1*mu[2])*np.exp(-.2*(1+.1*mu[1])*np.sqrt(.5*(x**2+y**2))) \
          - np.exp(.5*(np.cos(2*np.pi*(1+.1*mu[0])*x) + np.cos(2*np.pi*(1+.1*mu[0])*y))) \
          + 20 + np.exp(1)
    return u_0.reshape((1, u_0.shape[0]))

# Create linear space mesh
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"],
                            hp["y_min"], hp["y_max"], hp["n_y"])
np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))

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

#%% Sample the new model to generate a test prediction
mu_lhs = model.sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
X_v_tst, U_tst, _ = \
    model.create_snapshots(mu_lhs.shape[0], mu_lhs.shape[0], model.n_d, model.n_h, u, mu_lhs)
U_pred = model.predict(X_v_tst)
print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

#%% Samples graph
n_samples = 3
mu_lhs_in = model.sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
mu_lhs_out_min = model.sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
mu_lhs_out_max = model.sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))


# Contours for demo
n_plot_x = 1
n_plot_y = 2
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=3.0))
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
ax.legend()
ax.set_ylabel("$y$")
ax = fig.add_subplot(gs[0, 1])
U_pred = np.reshape(U_pred, (hp["n_x"], hp["n_y"], -1))
levels = list(range(2, 15))
ct = ax.contourf(xx, yy, U_pred.mean(-1), levels=levels, origin="lower")
plt.colorbar(ct)
ax.axis("equal")
ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend()
# plt.show()
savefig("cache/graph-means")

# Slices
n_plot_x = 2
n_plot_y = n_samples
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
    X_v_samples, U_samples, _ = \
        model.create_snapshots(mu_lhs.shape[0], mu_lhs.shape[0], model.n_d, model.n_h, u, mu_lhs)
    U_samples = np.reshape(U_samples, (hp["n_x"], hp["n_y"], -1))
                            
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
    for col, idx_i in enumerate(idx):
        lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
        X_i = X_v_samples[idx_i, :].reshape(1, -1)
        U_pred_i = model.restruct(model.predict(X_i))
        U_pred_i = np.reshape(U_pred_i, (hp["n_x"], hp["n_y"], -1))
        ax = fig.add_subplot(gs[row, col])
        ax.plot(x, U_pred_i[:, 199, 0], "C0-", label=r"$u_D(s_{" + lbl + r"})$")
        ax.plot(x, U_samples[:, 199, idx_i], "r--", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
        ax.set_xlabel("$x\ (y=0)$")
        if col == len(idx) - 1:
            ax.legend()
plt.tight_layout()
# plt.show()
savefig("cache/graph-samples")


# %%
