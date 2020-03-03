"""POD-NN modeling for 1D Shekel Equation."""
#%% Internal imports
import os
import sys
sys.path.append(os.path.join("..", ".."))
from podnn.plotting import figsize

#%% Imports
import numpy as np
import matplotlib.pyplot as plt

#%% Datagen
N_tst = 300
x_tst = np.linspace(-6, 6, N_tst).reshape(-1, 1)
D = 1
y_tst = x_tst**3
# D = 2
# u1_star = np.cos(x_star)
# u2_star = np.sin(x_star)
# u_star = np.column_stack((u1_star[:, 0], u2_star[:, 0]))

N = 200
lb = int(2/(2*6) * N_tst)
ub = int((2+2*4)/(2*6) * N_tst)
idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
# idx = np.array([26, 23, 4, 3, 27, 64, 58, 30, 18, 16, 2, 31, 65, 15, 11, 17, 57, 28, 34, 50])
x = x_tst[lb + idx]
y = y_tst[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 9
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])


#%% Datagen plot
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y_i$")
plt.plot(x_tst, y_tst, "r--", label=r"$u(x)$")
plt.legend()
# plt.show()
plt.savefig("review-bishop-toy-data.pdf")


#%%
d = 1
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y_i$")
plt.plot(x_tst, y_tst, "r--", label=r"$u(x)$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}_1(x)$")
plt.legend()
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}.pdf")

#%%
d = 3
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y_i$")
plt.plot(x_tst, y_tst, "r--", label=r"$u(x)$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}_1(x)$")
plt.legend()
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}.pdf")

#%%
d = 5
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y_i$")
plt.plot(x_tst, y_tst, "r--", label=r"$u(x)$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}_1(x)$")
plt.legend()
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}.pdf")

#%%
d = 10
w_star = np.polyfit(x.flatten(), y.flatten(), deg=d)
print(f"Weights w* for d={d}: {w_star}")
P_star = np.poly1d(w_star)
y_1_pred = P_star(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y_i$")
plt.plot(x_tst, y_tst, "r--", label=r"$u(x)$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}_1(x)$")
plt.legend()
plt.ylim((y_tst.min(), y_tst.max()))
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}-N200.pdf")

#%%
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
d = 10
reg = make_pipeline(PolynomialFeatures(d), Ridge(1e10))
reg.fit(x, y)
# print(f"Weights w* for d={d}: {reg.coef_}, {reg.intercept_}")
y_1_pred = reg.predict(x_tst)

#%%
fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.scatter(x, y, c="r", label=r"$y_i$")
plt.plot(x_tst, y_tst, "r--", label=r"$u(x)$")
plt.plot(x_tst, y_1_pred, "b-", label=r"$\hat{u}_1(x)$")
plt.legend()
plt.ylim((y_tst.min(), y_tst.max()))
# plt.show()
plt.savefig(f"review-bishop-toy-poly-{d}-l2-10e10.pdf")

#%%
import sys
import os
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras
dtype = "float64"
tf.keras.backend.set_floatx(dtype)

sys.path.append(os.path.join("..", ".."))
from podnn.plotting import figsize
#%% Model creation
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tfk.models.Sequential([
        # tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
        tfp.layers.VariableLayer(n, dtype=dtype, trainable=True),
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

model = tfk.models.Sequential([
    # tfk.layers.Dense(8, activation="linear"),
    # tfk.layers.Dense(2, activation="linear"),
    tfp.layers.DenseVariational(
        units=8,
        activation="relu",
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=1/N,
        dtype=dtype,
    ),
    tfp.layers.DenseVariational(
        units=2,
        activation="linear",
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=1/N,
        dtype=dtype,
    ),
    tfp.layers.DistributionLambda(lambda t:
        tfd.Normal(
            loc=t[..., :1],
            scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]),
        ),
        # tfd.Normal(
        #     loc=t,
        #     scale=1,
        # ),
    ),
])
lr = 0.01
model.compile(loss=lambda y, model: -model.log_prob(y), optimizer=tfk.optimizers.Adam(lr))
epochs = 12000
model.fit(x, y, epochs=epochs, verbose=0)

# yhat = model(x_tst)
# yhats = [model(x_tst) for _ in range(100)]
##%% Predictions and plotting

y_pred_list = []
y_pred_var_list = []
for i in range(200):
    yhat = model(x_tst)
    y_pred_list.append(yhat.mean().numpy())
    y_pred_var_list.append(yhat.variance().numpy())

u_pred = np.array(y_pred_list).mean(0)
u_pred_var = (np.array(y_pred_list)**2 + np.array(y_pred_var_list)).mean(0) - np.array(y_pred_list).mean(0) ** 2
u_pred_sig = np.sqrt(u_pred_var)

# y_preds = np.concatenate(y_pred_list, axis=1)
# u_pred = np.mean(y_preds, axis=1)
# u_pred_sig = np.std(y_preds, axis=1)

# u_pred = yhat.mean().numpy()
# u_pred_sig = yhat.stddev().numpy()
lower = u_pred - 2 * u_pred_sig
upper = u_pred + 2 * u_pred_sig

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_tst.ravel(), upper.ravel(), lower.ravel(), 
                    facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.plot(x_tst, u_pred, label=r"$\hat{u}_*(x)$")
plt.scatter(x, y, c="r", label=r"$u_T(x)$")
plt.plot(x_tst, y_tst, "r--", label=r"$u_*(x)$")
# for yhat in yhats:
#     plt.plot(x_tst, yhat.mean().numpy(), "b-", alpha=0.5)
#     plt.plot(x_tst, yhat.mean().numpy() + 2*yhat.stddev().numpy(), "b-", alpha=0.01)
#     plt.plot(x_tst, yhat.mean().numpy() - 2*yhat.stddev().numpy(), "b-", alpha=0.01)
plt.legend()
plt.xlabel("$x$")
plt.show()
# exit(0)
# # plt.savefig("results/gp.pdf")
# plt.savefig("results/cos.pdf")
# fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
# plt.fill_between(x_star[:, 0], lower[:, 1], upper[:, 1], 
#                     facecolor='orange', alpha=0.5, label=r"$2\sigma_{T,hf}(x)$")
# plt.plot(x_star, u_star[:, 1])
# plt.plot(x_star, u_pred[:, 1], "r--")
# plt.scatter(x_train, u_train[:, 1],)
# plt.savefig("results/sin.pdf")
