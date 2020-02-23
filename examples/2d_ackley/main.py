"""POD-NN modeling for 2D Ackley Equation."""
#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.logger import Logger, LoggerCallback
from podnn.metrics import re_s
from podnn.plotting import savefig, figsize

#%%
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfk = tf.keras
dtype = "float64"
tf.keras.backend.set_floatx(dtype)

#%% Prepare
from hyperparams import HP as hp
from hyperparams import u
print(hp)

resdir = "cache"
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"],
                            hp["y_min"], hp["y_max"], hp["n_y"])
np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
hp["n_L"] = 15
X_v_train, v_train, _, \
    X_v_val, v_val, U_val = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                                hp["n_s"],
                                                hp["train_val"],
                                                eps=hp["eps"], n_L=hp["n_L"],
                                                u_noise=hp["u_noise"],
                                                x_noise=hp["x_noise"])
N = hp["n_s"]
n_L = hp["n_L"]
#%% Model creation
def normalize(X):
    return (X - X_v_train.min(0)) / (X_v_train.max(0) - X_v_train.min(0))
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tfk.models.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
        # tfp.layers.VariableLayer(n, dtype=dtype, trainable=True),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1,
        ))
    ])

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = tf.math.log(tf.math.expm1(tf.constant(1., dtype=dtype)))
    return tfk.models.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(
                loc=t[..., :n],
                scale=1e-5 + 1. * tf.math.softplus(c + t[..., n:]),
            ),
            reinterpreted_batch_ndims=1,
        ))
    ])
klw = 1/(N)
bnn = tfk.models.Sequential([
    # tfk.layers.Dense(8, activation="linear"),
    # tfk.layers.Dense(2, activation="linear"),
    tfp.layers.DenseVariational(
        units=40,
        activation="relu",
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=klw,
        dtype=dtype,
    ),
    tfp.layers.DenseVariational(
        units=40,
        activation="relu",
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=klw,
        dtype=dtype,
    ),
    tfp.layers.DenseVariational(
        units=2 * n_L,
        activation="linear",
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=klw,
        dtype=dtype,
    ),
    tfp.layers.DistributionLambda(lambda t:
        tfd.MultivariateNormalDiag(
            loc=t[..., :n_L],
            scale_diag=1e-5 + tf.math.softplus(0.01 * t[..., n_L:]),
        ),
        # tfd.Normal(
        #     loc=t,
        #     scale=1,
        # ),
    ),
])
lr = 0.01
# bnn.compile(loss=lambda y, yhat: -tf.reduce_mean(yhat.log_prob(y)),
#               optimizer=tfk.optimizers.Adam(lr))
bnn.compile(loss=lambda y, yhat: -tf.reduce_sum(yhat.log_prob(y)),
              optimizer=tfk.optimizers.Adam(lr))
epochs = 10000
freq = 1000
logger = Logger(epochs, freq)
X_v_val_n = normalize(X_v_val)
logger.set_val_err_fn(lambda: {
    "RE_v": re_s(v_val.T, bnn(X_v_val).mean().numpy().T),
    "std": tf.reduce_sum(bnn(normalize(X_v_val)).mean().numpy().std(0)),
    })
bnn.fit(normalize(X_v_train), v_train, epochs=epochs,
          verbose=0, callbacks=[LoggerCallback(logger)])

model.regnn = bnn
n_h = U_val.shape[0]

#%%
def predict_v(X, samples=20):
    yhat = bnn(normalize(X))
    v_pred = np.array([yhat.mean().numpy() for _ in range(samples)]).mean(0)
    return v_pred, np.zeros_like(v_pred)
def predict(X, samples=20):
    U_pred_samples = np.zeros((model.n_h, X.shape[0], samples))
    U_pred_sig_samples = np.zeros((model.n_h, X.shape[0], samples))

    for i in range(samples):
        v_dist = bnn(normalize(X))
        v_pred, v_pred_var = v_dist.mean().numpy(), v_dist.variance().numpy()
        U_pred_samples[:, :, i] = model.project_to_U(v_pred)
        U_pred_sig_samples[:, :, i] = model.project_to_U(np.sqrt(v_pred_var))

    U_pred = U_pred_samples.mean(-1)
    U_pred_var = (U_pred_sig_samples**2 + U_pred_samples ** 2).mean(-1) - U_pred ** 2
    U_pred_sig = np.sqrt(U_pred_var)

    return U_pred.astype(model.dtype), U_pred_sig.astype(model.dtype)

v_pred, _ = predict_v(X_v_val)
U_pred = model.project_to_U(v_pred)

err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")
err_val = re_s(v_val.T, v_pred.T)
print(f"RE_v: {err_val:4f}")
#%%
yhat = bnn(normalize(X_v_val))
for i in [0, 1]:
    plt.plot(yhat[i].numpy(), "b-")
    plt.plot(v_val[i], "r--")
plt.show()

#%%
# y_pred_list = []
# y_pred_var_list = []
# for i in range(200):
#     yhat = model(x_tst)
#     y_pred_list.append(yhat.mean().numpy())
#     y_pred_var_list.append(yhat.variance().numpy())

# u_pred = np.array(y_pred_list).mean(0)
# u_pred_var = (np.array(y_pred_list)**2 + np.array(y_pred_var_list)).mean(0) - np.array(y_pred_list).mean(0) ** 2
# u_pred_sig = np.sqrt(u_pred_var)
#%% Train
# model.initBNN(hp["h_layers"],
#                 hp["lr"], 1/X_v_train.shape[0],
#                 hp["norm"])
# train_res = model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"],
#                         freq=hp["log_frequency"],
#                         silent=False)

#%% Validation metrics
# U_pred, _ = model.predict(X_v_val, samples=100)
# err_val = re_s(U_val, U_pred)
# print(f"RE_v: {err_val:4f}")

# import matplotlib.pyplot as plt
# v_pred, v_pred_sig = model.predict_v(X_v_val)
# # print(v_pred)
# # dist = model.regnn.model(X_v_val)
# # v_pred, v_pred_sig = dist.mean().numpy(), np.sqrt(dist.variance())
# print(v_pred.shape)
# x = np.arange(v_pred.shape[1])
# plt.plot(x, v_pred[0])
# plt.plot(x, v_val[0])
# plt.fill_between(x, v_pred[0] - 2*v_pred_sig[0],
#                     v_pred[0] + 2*v_pred_sig[0], alpha=0.3)
# plt.show()

# # U_pred = model.restruct(U_pred)
# # U_val = model.restruct(U_val)
# x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
# # lower = U_pred - 3 * U_pred_sig
# # upper = U_pred + 3 * U_pred_sig
# # plt.fill_between(x, lower[:, 0], upper[:, 0], 
# #                     facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
# plt.plot(x, U_pred[0], "b-")
# plt.plot(x, U_val[0], "r--")
# # plt.plot(x, model.predict(X_v_test)[:, 0])
# plt.show()

#%% Sample the new model to generate a test prediction
mu_lhs = model.sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
X_v_tst, U_tst, _, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
U_pred, U_pred_sig = predict(X_v_tst)
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
plt.show()
# savefig("results/graph-means")

#%% Slices
n_plot_x = 2
n_plot_y = n_samples
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
    X_v_samples, U_samples, _, _ = \
        model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
    U_samples = np.reshape(U_samples, (hp["n_x"], hp["n_y"], -1))
                            
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
    for col, idx_i in enumerate(idx):
        lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
        X_i = X_v_samples[idx_i, :].reshape(1, -1)
        U_pred_i, U_pred_i_sig = predict(X_i)
        U_pred_i = np.reshape(U_pred_i, (hp["n_x"], hp["n_y"], -1))
        U_pred_i_sig = np.reshape(U_pred_i_sig, (hp["n_x"], hp["n_y"], -1))
        ax = fig.add_subplot(gs[row, col])
        ax.plot(x, U_pred_i[:, 199, 0], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
        ax.plot(x, U_samples[:, 199, idx_i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
        lower = U_pred_i[:, 199, 0] - 3*U_pred_i_sig[:, 199, 0]
        upper = U_pred_i[:, 199, 0] + 3*U_pred_i_sig[:, 199, 0]
        ax.fill_between(x, lower, upper, alpha=0.2, label=r"$3\sigma_D(s_{" + lbl + r"})$")
        ax.set_xlabel("$x\ (y=0)$")
        if col == len(idx) - 1:
            ax.legend()
plt.tight_layout()
plt.show()
# savefig("results/graph-samples")


# %%
