"""POD-NN modeling for 1D Shekel Equation."""
#%%

import sys
import os
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.plotting import genresultdir

from podnn.varneuralnetwork import VarNeuralNetwork
from podnn.metrics import re_mean_std, re_max
from podnn.mesh import create_linear_mesh
from podnn.logger import Logger
from podnn.advneuralnetwork import NORM_MEANSTD, NORM_NONE
from podnn.plotting import figsize

#%% Plot function
def plot(x, y, x_tst, y_tst, yhat=None, yhats=[], lower=None, upper=None):
    fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
    if lower is not None and upper is not None:
        plt.fill_between(x_tst[:, 0], lower[:, 0], upper[:, 0], 
                            facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
    # plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
    plt.scatter(x, y[:, 0], c="r", label=r"$u_T(x)$")
    plt.plot(x_tst, y_tst[:, 0], "r--", label=r"$u_*(x)$")
    if yhat is not None:
        plt.plot(x_tst, yhat, "C0", label=r"$\hat{u}_*(x)$")
    for yhat_i in yhats:
        plt.plot(x_tst, yhat_i, "C0", alpha=0.1)
    plt.legend()
    plt.xlabel("$x$")
    plt.show()

#%% Datagen
N_star = 100
D = 1
x_tst = np.linspace(-6, 6, N_star).reshape((N_star, 1))
y_tst = x_tst**3

#%% Training split
N = 20
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
x = x_tst[lb + idx]
y = y_tst[lb + idx]
noise_std = 3
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])

#%% Loss
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

#%% Layers
layers = [20, 20, 1]

#%% Case 1: no uncertainty, linear reg
# Build model.
model = tfk.Sequential([
  tfk.layers.Dense(20, activation="relu"),
  tfk.layers.Dense(20, activation="relu"),
  tfk.layers.Dense(1),
  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x, y, epochs=500, verbose=False)

# Make predictions.
yhat = model(x_tst)
plot(x, y, x_tst, y_tst, yhat.mean())

#%% Case 2: known unknowns (alatoric uncertainty)
# Build model.
model = tfk.Sequential([
    tfk.layers.Dense(20, activation="relu"),
    tfk.layers.Dense(20, activation="relu"),
    tfk.layers.Dense(1 + 1),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
])


# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x, y, epochs=500, verbose=False)

# Make predictions.
yhat = model(x_tst)
lower = yhat.mean() - 2 * yhat.stddev()
upper = yhat.mean() + 2 * yhat.stddev()
plot(x, y, x_tst, y_tst, yhat.mean(), lower=lower, upper=upper)

#%% Case 3: unknown unknowns (epistemic uncertainty)
# Build model.

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])
# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])
model = tfk.Sequential([
    tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable,
                                activation=tf.nn.relu),
    tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable,
                                activation=tf.nn.relu),
    tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable,
                                activation=None),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])


# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x, y, epochs=500, verbose=True)

# Make predictions.
yhats = np.array([model(x_tst).mean() for _ in range(100)])
plot(x, y, x_tst, y_tst, yhats.mean(0), yhats=yhats)

#%% Case 4: both unknowns
# Build model.
model = tf.keras.Sequential([
#   tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable, kl_weight=1/x.shape[0],
#                               activation=tf.nn.relu),
#   tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable, kl_weight=1/x.shape[0],
#                               activation=tf.nn.relu),
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/x.shape[0]),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])

def normalize(x_to_be_n):
    return (x_to_be_n - x.mean()) / x.std()

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
model.fit(normalize(x), y, epochs=2000, verbose=1)

# Profit.
yhats = [model(normalize(x_tst)) for _ in range(500)]
yhats_mean = np.array([yh.mean() for yh in yhats])
yhats_var = np.array([yh.variance() for yh in yhats])
print(yhats_mean.shape, yhats_var.shape)
yhat = yhats_mean.mean(0)
yhat_var = (yhats_var + yhat ** 2).mean(0) - yhat ** 2
print(yhat.shape, yhat_var.shape)
lower = yhat - 3 * np.sqrt(yhat_var)
upper = yhat + 3 * np.sqrt(yhat_var)
plot(x, y, x_tst, y_tst, yhat, lower=lower, upper=upper)
# yhats = np.array([model(x_tst).mean() for _ in range(100)])
# plot(x, y, x_tst, y_tst, yhats.mean(0), yhats=yhats)
# # Make predictions.
# yhats = np.array([model(x_tst).mean() for _ in range(100)])
# yhat = yhats.mean(0)
# yhat_var = (yhats_var + yhats ** 2).mean(0) - yhat ** 2 
# lower = yhat - 2 * np.sqrt(yhat_var)
# upper = yhat + 2 * np.sqrt(yhat_var)

# plot(x, y, x_tst, y_tst, yhat, lower=lower, upper=upper)


#%% Plot
u_pred = u_pred_samples.mean(-1)
u_pred_var = (u_pred_var_samples + u_pred_samples ** 2).mean(-1) - u_pred ** 2
lower = u_pred - 3 * np.sqrt(u_pred_var)
upper = u_pred + 3 * np.sqrt(u_pred_var)
print(u_pred.shape, u_pred_var.shape)