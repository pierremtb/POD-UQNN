"""POD-NN modeling for the 1D time-dep Burgers Equation."""

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

def u(X, t, mu):
    """Burgers2 explicit solution."""
    x = X[0]
    mu = mu[0]

    if t == 1.:
        res = x / (1 + np.exp(1/(4*mu)*(x**2 - 1/4)))
    else:
        t0 = np.exp(1 / (8*mu))
        res = (x/t) / (1 + np.sqrt(t/t0)*np.exp(x**2/(4*mu*t)))
    return res.reshape((1, x.shape[0]))

# Create linear space mesh
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
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
                                   hp["eps"], eps_init=hp["eps_init"],
                                   t_min=hp["t_min"], t_max=hp["t_max"])

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

import sys
import os
import yaml
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_mean_std
from podnn.mesh import create_linear_mesh

from genhifi import u, generate_test_dataset
from plot import plot_results


def main(hp, gen_test=False, use_cached_dataset=False,
         no_plot=False):
    """Full example to run POD-NN on 1dt_burger."""

    if gen_test:
        generate_test_dataset()

    if not use_cached_dataset:
        # Create linear space mesh
        x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
        np.save(os.path.join("cache", "x_mesh.npy"), x_mesh)
    else:
        x_mesh = np.load(os.path.join("cache", "x_mesh.npy"))

    # Init the model
    model = PodnnModel("cache", hp["n_v"], x_mesh, hp["n_t"])

    # Generate the dataset from the mesh and params
    X_v_train, v_train, \
        X_v_test, _, \
        U_test = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                        hp["n_s"],
                                        hp["train_val"],
                                        hp["eps"], eps_init=hp["eps_init"],
                                        t_min=hp["t_min"], t_max=hp["t_max"],
                                        use_cache=use_cached_dataset)

    # Train
    model.initNN(hp["h_layers"], hp["lr"], hp["lambda"])
    train_res = model.train(X_v_train, v_train, hp["epochs"],
                            hp["train_val"], freq=hp["log_frequency"])

    # Predict and restruct
    U_pred = model.predict(X_v_test)
    U_pred = model.restruct(U_pred)
    U_test = model.restruct(U_test)

    # Compute relative error
    error_test_mean, error_test_std = re_mean_std(U_test, U_pred)
    print(f"Test relative error: mean {error_test_mean:4f}, std {error_test_std:4f}")

    # Sample the new model to generate a HiFi prediction
    n_s_tst = hp["n_s_tst"]
    print("Sampling {n_s_tst} parameters")
    X_v_test_hifi = model.generate_hifi_inputs(n_s_tst, hp["mu_min"], hp["mu_max"],
                                               hp["t_min"], hp["t_max"])
    print("Predicting the {n_s_tst} corresponding solutions")
    U_pred_hifi_mean, U_pred_hifi_std = model.predict_heavy(X_v_test_hifi)
    U_pred_hifi_mean = model.restruct(U_pred_hifi_mean, no_s=True)
    U_pred_hifi_std = model.restruct(U_pred_hifi_std, no_s=True)

    # Plot against test and save
    return plot_results(U_pred, U_pred_hifi_mean, U_pred_hifi_std,
                        train_res, hp, no_plot)

if __name__ == "__main__":
    # Custom hyperparameters as command-line arg
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as HPFile:
            HP = yaml.load(HPFile)
    # Default ones
    else:
        from hyperparams import HP

    main(HP, gen_test=False, use_cached_dataset=False)
    # main(HP, gen_test=False, use_cached_dataset=True)
