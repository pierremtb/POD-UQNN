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

def normalize(x, mean, std):
    return (x - mean) / std

#%% Datagen
N_star = 300
D = 1
x_tst = np.linspace(-6, 6, N_star).reshape((N_star, 1))
y_tst = x_tst**3

#%% Training split
N = 20
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
# idx = np.array([26, 23,  4,  3, 27, 64, 58, 30, 18, 16,  2, 31, 65, 15, 11, 17, 57, 28, 34, 50])
x = x_tst[lb + idx]
x_range = [x.min(), x.max()]
y = y_tst[lb + idx]
noise_std = 3
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])

layers = [1, 20, 20, 1]
# layers = [1, 1]
dtype = "float64"
activation = "relu"
# activation = "softplus"
tf.keras.backend.set_floatx(dtype)

mean = x.mean()
std =  x.std()
x_n = normalize(x, mean, std)
x_tst_n = normalize(x_tst, mean, std)

#%% Loss
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

#%% Case 0: no uncertainty
print("No uncertainty: regular NN regression")
# Build model.
model = tfk.Sequential([
    tfk.layers.InputLayer((layers[0],)),
    *[tfk.layers.Dense(width, activation) for width in layers[1:-1]],
    tfk.layers.Dense(layers[-1] * 2),
  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x_n, y, epochs=500, verbose=False)

# Make predictions.
yhat = model(x_tst_n)
plot(x, y, x_tst, y_tst, yhat.mean())

#%% Case 1: known unknowns (aleatoric uncertainty)
print("Built-in variance: known unknowns (aleatoric)")
# Build model.
model = tfk.Sequential([
    tfk.layers.InputLayer((layers[0],)),
    *[tfk.layers.Dense(width, activation) for width in layers[1:-1]],
    tfk.layers.Dense(layers[-1] * 2),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
])


# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x_n, y, epochs=1500, verbose=False)

# Make predictions.
yhat = model(x_tst_n)
lower = yhat.mean() - 2 * yhat.stddev()
upper = yhat.mean() + 2 * yhat.stddev()
plot(x, y, x_tst, y_tst, yhat.mean(), lower=lower, upper=upper)

#%% Case 2: unknown unknowns (epistemic uncertainty)
print("Variational inference on layers: unknown unknowns (epistemic)")
# # Build model.

# # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
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
    tfk.layers.InputLayer((layers[0],)),
    *[tfp.layers.DenseVariational(width, posterior_mean_field, prior_trainable, activation=activation)
      for width in layers[1:-1]],
    tfp.layers.DenseVariational(layers[-1], posterior_mean_field, prior_trainable),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])


# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x_n, y, batch_size=None, epochs=1500, verbose=False)

# Make predictions.
yhats = np.array([model(x_tst_n).mean() for _ in range(200)])
yhat = yhats.mean(0)
yhat_var = yhats.var(0)
# yhats = [model(x_tst_n) for _ in range(100)]
# yhats_mean = np.array([yh.mean() for yh in yhats])
# yhats_var = np.array([yh.variance() for yh in yhats])
# yhat = yhats_mean.mean(0)
# yhat_var = (yhats_var + yhat ** 2).mean(0) - yhat ** 2

lower = yhat - 2 * np.sqrt(yhat_var)
upper = yhat + 2 * np.sqrt(yhat_var)
plot(x, y, x_tst, y_tst, yhat, lower=lower, upper=upper)

#%% Case 3: known and unknown unknowns
print("Built-in var and varitional layers: aleatoric and epistemic")
model = tfk.Sequential([
    tfk.layers.InputLayer((layers[0],)),
    *[tfp.layers.DenseVariational(width, posterior_mean_field, prior_trainable, activation=activation)
      for width in layers[1:-1]],
    tfp.layers.DenseVariational(layers[-1] * 2, posterior_mean_field, prior_trainable),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
])


# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_epoch_end(self, epoch, logs=None):
        if (epoch % 100 == 0):
            print(epoch, logs["loss"])
logger = tfk.callbacks.ProgbarLogger(count_mode="samples")
model.fit(x_n, y, batch_size=None, epochs=30000, verbose=0, callbacks=[CustomCallback()])

#%%
# Make predictions.
yhats = [model(x_tst_n) for _ in range(100)]
yhats_mean = np.array([p.mean() for p in yhats])
# yhat = yhats_mean.mean(0)
# yhat_var = yhats_mean.var(0)
yhats_var = np.array([p.variance() for p in yhats])
yhat = yhats_mean.mean(0)
yhat_var = (yhats_var + yhat ** 2).mean(0) - yhat ** 2

lower = yhat - 2 * np.sqrt(yhat_var)
upper = yhat + 2 * np.sqrt(yhat_var)
# plot(x, y, x_tst, y_tst, yhat, lower=lower, upper=upper)
plot(x, y, x_tst, y_tst, yhat, yhats=yhats_mean[:10], lower=lower, upper=upper)
