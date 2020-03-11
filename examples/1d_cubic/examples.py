"""POD-NN modeling for 1D Shekel Equation."""
#%% Internal imports
import os
import sys
sys.path.append(os.path.join("..", ".."))
from podnn.plotting import figsize, savefig

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

#%% Datagen
N_tst = 300
x_tst = np.linspace(-6, 6, N_tst).reshape(-1, 1)
D = 1
y_tst = x_tst**3

N = 20
lb = int(2/(2*6) * N_tst)
ub = int((2+2*4)/(2*6) * N_tst)
# idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        91, 100, 188,   8,  63,  98,  78])
x = x_tst[lb + idx]
y = y_tst[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 9
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])

#%% Datagen plot
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.legend()
# plt.show()
plt.savefig("review-bishop-toy-data.pdf", bbox_inches='tight', pad_inches=0)

#%%
d = 1
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}(x_\textrm{tst})$")
plt.legend()
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}.pdf", bbox_inches='tight', pad_inches=0)

#%%
d = 3
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}(x_\textrm{tst})$")
plt.legend()
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}.pdf", bbox_inches='tight', pad_inches=0)

#%%
d = 10
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}(x_\textrm{tst})$")
plt.legend()
plt.ylim((y_tst.min(), y_tst.max()))
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}.pdf", bbox_inches='tight', pad_inches=0)

#%%
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
d = 10
alpha = 0.45
reg = make_pipeline(PolynomialFeatures(d), Ridge(alpha=alpha))
reg.fit(x, y)
y_1_pred = reg.predict(x_tst)

##%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}(x_\textrm{tst})$")
plt.legend()
plt.ylim((y_tst.min(), y_tst.max()))
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}-l2-10e10.pdf", bbox_inches='tight', pad_inches=0)

#%%
N = 200
lb = int(2/(2*6) * N_tst)
ub = int((2+2*4)/(2*6) * N_tst)
idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
# idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        # 91, 100, 188,   8,  63,  98,  78])
x = x_tst[lb + idx]
y = y_tst[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 9
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])
d = 10
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}(x_\textrm{tst})$")
plt.legend()
plt.ylim((y_tst.min(), y_tst.max()))
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}-N{N}.pdf", bbox_inches='tight', pad_inches=0)

#%%
N = 20
lb = int(2/(2*6) * N_tst)
ub = int((2+2*4)/(2*6) * N_tst)
# idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        91, 100, 188,   8,  63,  98,  78])
x = x_tst[lb + idx]
y = y_tst[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 9
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])
#%%
d = 3
beta = 1/noise_std**2
# alpha = 5e-3
alpha = 0.11

# Polynomial basis
def phi(x_):
    return np.array([x_**i for i in range(d+1)])

# S matrix
S_inv = alpha * np.identity(d+1) + \
        beta * np.array([phi(x_i).dot(phi(x_i).T) for x_i in x]) \
                 .sum(0)
S = np.linalg.inv(S_inv)

# Mean for a given x
def mean(x_):
    sum_n = np.array([phi(x[i]) * y[i] for i in range(N)]) \
              .sum(0)
    return beta * phi(x_).T.dot(S).dot(sum_n)

# Variance for a given x
def variance(x_):
    return 1/beta + phi(x_).T.dot(S).dot(phi(x_))

# Predictions
f_post = np.zeros_like(x_tst)
var = np.zeros_like(x_tst)
for i, x_i in enumerate(x_tst):
    f_post[i] = mean(x_i)
    var[i] = variance(x_i)
sig = np.sqrt(var)

upper = f_post + 2 * sig
lower = f_post - 2 * sig

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_tst.ravel(), upper.ravel(), lower.ravel(),
                    facecolor='C0', alpha=0.3, label=r"$2\sigma(x_\textrm{tst})$")
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, f_post, "b-", label=r"$\hat{u}(x_\textrm{tst})$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.legend()
plt.ylim((y_tst.min(), y_tst.max()))
plt.savefig(f"review-bishop-toy-polybayes-{d}.pdf", bbox_inches='tight', pad_inches=0)

#%%
import tensorflow as tf
tfk = tf.keras

model = tfk.Sequential([
    tfk.layers.Dense(5,
        activation="relu",
        kernel_regularizer=tfk.regularizers.l2(0.001)),
    tfk.layers.Dense(1),
])
model.compile(optimizer=tfk.optimizers.Adam(0.1), loss="mse")
model.fit(x, y, epochs=1000, verbose=0)
f_post = model.predict(x_tst)

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y$")
plt.plot(x_tst, f_post, "b-", label=r"$\hat{u}(x_\textrm{tst})$")
plt.plot(x_tst, y_tst, "r--", label=r"$y_\textrm{tst}$")
plt.legend()
plt.ylim((y_tst.min(), y_tst.max()))
plt.savefig(f"review-bishop-toy-nn.pdf", bbox_inches='tight', pad_inches=0)

#%%

# Datagen
N_tst = 50
x_tst = np.linspace(-5, 5, N_tst).reshape(-1,1)
y_tst = np.sin(x_tst)
x = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
y = np.sin(x)

# Kernel and covariance
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)
param = 0.1
K_ss = kernel(x_tst, x_tst, param)

# Get cholesky decomposition of covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(N_tst))

# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(N_tst,3)))

# Apply the kernel function to our training points
K = kernel(x, x, param)
L = np.linalg.cholesky(K + 0.00005*np.eye(len(x)))

# Compute the mean at our test points and standard deviation.
K_s = kernel(x, x_tst, param)
Lk = np.linalg.solve(L, K_s)
mean = np.dot(Lk.T, np.linalg.solve(L, y)).reshape((N_tst,))
var = np.diag(K_ss) - np.sum(Lk**2, axis=0)
sig = np.sqrt(var)

# Draw 3 samples from the posterior
L = np.linalg.cholesky(K_ss + 1e-6*np.eye(N_tst) - np.dot(Lk.T, Lk))
f_post = mean.reshape(-1,1) + np.dot(L, np.random.normal(size=(N_tst,3)))

lower = mean - 2 * sig
upper = mean + 2 * sig
fig = plt.figure(figsize=figsize(2, 2, scale=2.5))
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, :])
plt.axis([-5, 5, -3, 3])
ax.plot(x_tst, f_prior)
ax.set_title("Prior samples")
ax.set_xlabel("$x$")
ax = fig.add_subplot(gs[1, :])
plt.fill_between(x_tst.ravel(), upper.ravel(), lower.ravel(), 
                    facecolor='C0', alpha=0.2, label=r"$2\sigma_{T}(x)$")
ax.scatter(x, y, c="r", label=r"$y_i$")
ax.plot(x_tst, f_post, "b-", alpha=0.3)
ax.plot(x_tst, mean, "b-", label=r"$\hat{u}(x)$")
ax.plot(x_tst, y_tst, "r--", label=r"$u(x)$")
ax.set_title("Observed data and posterior samples")
ax.set_xlabel("$x$")
ax.legend()
plt.tight_layout()
plt.savefig(f"review-piml-gp.pdf", bbox_inches='tight', pad_inches=0)

#%%

N_tst = 300
x_tst = np.linspace(-6, 6, N_tst).reshape(-1, 1)
D = 1
y_tst = x_tst**3

N = 20
lb = int(2/(2*6) * N_tst)
ub = int((2+2*4)/(2*6) * N_tst)
# idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        91, 100, 188,   8,  63,  98,  78])
x = x_tst[lb + idx]
y = y_tst[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 9
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])

#%%
from podnn.custombnn import BayesianNeuralNetwork
layers = [1, 20, 20, 1]
batch_size = N
num_batches = N / batch_size
klw = 1.0 / num_batches
model = BayesianNeuralNetwork(layers, lr=0.05, klw=klw, soft_0=0.1,
                              sigma_alea=noise_std,
                              adv_eps=None, norm="minmax")
model.fit(x, y, epochs=15000, batch_size=batch_size)
u_pred, u_pred_var = model.predict(x_tst) 
u_pred_sig = np.sqrt(u_pred_var)

#%% Predictions and plotting
lower = u_pred - 2 * u_pred_sig
upper = u_pred + 2 * u_pred_sig

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_tst.ravel(), upper.ravel(), lower.ravel(), 
                    facecolor='C0', alpha=0.3, label=r"$2\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.plot(x_tst, u_pred, label=r"$\hat{u}_*(x)$")
plt.scatter(x, y, c="r", label=r"$u_T(x)$")
plt.plot(x_tst, y_tst, "r--", label=r"$u_*(x)$")
plt.ylim((y_tst.min(), y_tst.max()))
plt.xlabel("$x$")
plt.legend()
plt.tight_layout()
plt.savefig(f"uq-toy-bnn.pdf", bbox_inches='tight', pad_inches=0)
#%%
# import sys
# import os
# import tensorflow as tf
# import tensorflow_probability as tfp

# tfd = tfp.distributions
# tfk = tf.keras
# dtype = "float64"
# tf.keras.backend.set_floatx(dtype)

# sys.path.append(os.path.join("..", ".."))
# from podnn.plotting import figsize
# #%% Model creation
# def prior_trainable(kernel_size, bias_size=0, dtype=None):
#     n = kernel_size + bias_size
#     return tfk.models.Sequential([
#         # tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
#         tfp.layers.VariableLayer(n, dtype=dtype, trainable=True),
#         tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#             tfd.Normal(loc=t, scale=1),
#             reinterpreted_batch_ndims=1,
#         ))
#     ])

# def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
#     n = kernel_size + bias_size
#     c = tf.math.log(tf.math.expm1(tf.constant(1., dtype=dtype)))
#     return tfk.models.Sequential([
#         tfp.layers.VariableLayer(2 * n, dtype=dtype),
#         tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#             tfd.Normal(
#                 loc=t[..., :n],
#                 scale=1e-5 + 1. * tf.math.softplus(c + t[..., n:]),
#             ),
#             reinterpreted_batch_ndims=1,
#         ))
#     ])

# model = tfk.models.Sequential([
#     # tfk.layers.Dense(8, activation="linear"),
#     # tfk.layers.Dense(2, activation="linear"),
#     tfp.layers.DenseVariational(
#         units=8,
#         activation="relu",
#         make_posterior_fn=posterior_mean_field,
#         make_prior_fn=prior_trainable,
#         kl_weight=1/N,
#         dtype=dtype,
#     ),
#     tfp.layers.DenseVariational(
#         units=2,
#         activation="linear",
#         make_posterior_fn=posterior_mean_field,
#         make_prior_fn=prior_trainable,
#         kl_weight=1/N,
#         dtype=dtype,
#     ),
#     tfp.layers.DistributionLambda(lambda t:
#         tfd.Normal(
#             loc=t[..., :1],
#             scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]),
#         ),
#         # tfd.Normal(
#         #     loc=t,
#         #     scale=1,
#         # ),
#     ),
# ])
# lr = 0.01
# model.compile(loss=lambda y, model: -model.log_prob(y), optimizer=tfk.optimizers.Adam(lr))
# epochs = 12000
# model.fit(x, y, epochs=epochs, verbose=0)

# # yhat = model(x_tst)
# # yhats = [model(x_tst) for _ in range(100)]
# ##%% Predictions and plotting

# y_pred_list = []
# y_pred_var_list = []
# for i in range(200):
#     yhat = model(x_tst)
#     y_pred_list.append(yhat.mean().numpy())
#     y_pred_var_list.append(yhat.variance().numpy())

# u_pred = np.array(y_pred_list).mean(0)
# u_pred_var = (np.array(y_pred_list)**2 + np.array(y_pred_var_list)).mean(0) - np.array(y_pred_list).mean(0) ** 2
# u_pred_sig = np.sqrt(u_pred_var)

# # y_preds = np.concatenate(y_pred_list, axis=1)
# # u_pred = np.mean(y_preds, axis=1)
# # u_pred_sig = np.std(y_preds, axis=1)

# # u_pred = yhat.mean().numpy()
# # u_pred_sig = yhat.stddev().numpy()
# lower = u_pred - 2 * u_pred_sig
# upper = u_pred + 2 * u_pred_sig

# fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
# plt.fill_between(x_tst.ravel(), upper.ravel(), lower.ravel(), 
#                     facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
# # plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
# plt.plot(x_tst, u_pred, label=r"$\hat{u}_*(x)$")
# plt.scatter(x, y, c="r", label=r"$u_T(x)$")
# plt.plot(x_tst, y_tst, "r--", label=r"$u_*(x)$")
# # for yhat in yhats:
# #     plt.plot(x_tst, yhat.mean().numpy(), "b-", alpha=0.5)
# #     plt.plot(x_tst, yhat.mean().numpy() + 2*yhat.stddev().numpy(), "b-", alpha=0.01)
# #     plt.plot(x_tst, yhat.mean().numpy() - 2*yhat.stddev().numpy(), "b-", alpha=0.01)
# plt.legend()
# plt.xlabel("$x$")
# plt.show()
# # exit(0)
# # # plt.savefig("results/gp.pdf")
# # plt.savefig("results/cos.pdf")
# # fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
# # plt.fill_between(x_star[:, 0], lower[:, 1], upper[:, 1], 
# #                     facecolor='orange', alpha=0.5, label=r"$2\sigma_{T,hf}(x)$")
# # plt.plot(x_star, u_star[:, 1])
# # plt.plot(x_star, u_pred[:, 1], "r--")
# # plt.scatter(x_train, u_train[:, 1],)
# # plt.savefig("results/sin.pdf")
