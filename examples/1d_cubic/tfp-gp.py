"""POD-NN modeling for 1D Shekel Equation."""

import sys
import os
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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

# Datagen
N_star = 100
D = 1
x_star = np.linspace(-6, 6, N_star).reshape((N_star, 1))
u_star = x_star**3
# D = 1
# u1_star = np.cos(x_star)
# u2_star = np.sin(x_star)
# u_star = np.column_stack((u1_star[:, 0], u2_star[:, 0]))

N = 20
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 3
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

"""Well not only is it possible, but this colab shows how! (In context of linear regression problems.)"""

#@title Synthesize dataset.
w0 = 0.125
b0 = 5.
x_range = [x_train.min(), x_train.max()]

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')

# Build model.
num_inducing_points = int(N / 2)

class RBFKernelFn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
                initializer=tf.constant_initializer(0),
                dtype=dtype,
                name='amplitude')

        self._length_scale = self.add_variable(
                initializer=tf.constant_initializer(0),
                dtype=dtype,
                name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5. * self._length_scale)
        )

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[1]),
    tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
    tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(),
        event_shape=[1],
        inducing_index_points_initializer=tf.constant_initializer(
            np.linspace(*x_range, num=num_inducing_points,
                        dtype=x_train.dtype)[..., np.newaxis]),
        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(np.array(0.54).astype(x_train.dtype))),
    ),
])

# Do inference.
batch_size = N
loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size, x_train.dtype) / x_train.shape[0])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=loss)
model.fit(x_train, u_train, batch_size=batch_size, epochs=1000, verbose=False)

# Profit.
yhat = model(x_star)
assert isinstance(yhat, tfd.Distribution)

plt.figure(figsize=[6, 1.5])  # inches
plt.plot(x_train, u_train, 'r.', label='observed')
plt.plot(x_star, u_star, 'r--', label='observed')
plt.plot(x_star, yhat.mean(), 'b-', label='observed')
num_samples = 500
samples = np.zeros((num_samples, *u_star.shape))
for i in range(num_samples):
  samples[i] = yhat.sample().numpy()
  if i % 100 == 0:
    plt.plot(x_star,
            samples[i],
            'C0', alpha=0.5,
            linewidth=0.9,
            label='ensemble means' if i == 0 else None)
stddev = samples.std(0)
lower = yhat.mean().numpy() - 3 * stddev
upper = yhat.mean().numpy() + 3 * stddev
print(lower.shape)
plt.fill_between(x_star[:, 0], lower[:, 0], upper[:, 0],
                 facecolor='C0', alpha=0.3)

plt.show()

exit(0)

def neg_log_lik(y, rv_y):
    """Evaluate negative log-likelihood of a random variable `rv_y` for data `y`"""
    return -rv_y.log_prob(y)

# model outputs normal distribution with constant variance
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t, scale=1.0)
    )
])

# train the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), 
                    loss=neg_log_lik)
model.fit(x_train, u_train, 
                 epochs=5000,
                 verbose=True)

print(f"predicted w : {model.layers[-2].kernel.numpy()}")
print(f"predicted b : {model.layers[-2].bias.numpy()}")

# Make predictions.
M = 100
yhats = [model(x_star) for _ in range(M)]
u_pred_samples = np.zeros((N_star, D, M))
u_pred_var_samples = np.zeros((N_star, D, M))
for i, yhat in enumerate(yhats):
    u_pred_samples[i] = yhat.mean().numpy().reshape(1, -1)
    u_pred_var_samples[i] = yhat.stddev().numpy().reshape(1, -1)

# Model creation
# def gen_and_train_model():
#     layers = [1, 50, 50, D]
#     epochs = 20000
#     lr = 0.0001
#     model = VarNeuralNetwork(layers, lr, 1e-10, lb=x_train.mean(), ub=x_train.std())
#     logger = Logger(epochs, 5000, silent=True)
#     logger.set_val_err_fn(lambda _: {})

#     # Training
#     model.fit(x_train, u_train, epochs, logger)

#     # Make the prediction on the meshed x-axis (ask for MSE as well)
#     return model.predict(x_star)

# M = 5
# u_pred_samples = np.zeros((N_star, D, M))
# u_pred_var_samples = np.zeros((N_star, D, M))
# for i in range(0, M):
#     print(f"\nTraining model {i + 1}/{M}...\n")
#     u_pred_samples[:, :, i], u_pred_var_samples[:, :, i] = gen_and_train_model()

u_pred = u_pred_samples.mean(-1)
u_pred_var = (u_pred_var_samples + u_pred_samples ** 2).mean(-1) - u_pred ** 2
lower = u_pred - 3 * np.sqrt(u_pred_var)
upper = u_pred + 3 * np.sqrt(u_pred_var)
print(u_pred.shape, u_pred_var.shape)

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_star[:, 0], lower[:, 0], upper[:, 0], 
                    facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.scatter(x_train, u_train[:, 0], c="r", label=r"$u_T(x)$")
plt.plot(x_star, u_star[:, 0], "r--", label=r"$u_*(x)$")
plt.plot(x_star, u_pred[:, 0], label=r"$\hat{u}_*(x)$")
plt.legend()
plt.xlabel("$x$")
plt.show()
# plt.savefig("results/gp.pdf")
# plt.savefig("results/cos.pdf")
# plt.savefig("results/bnn.pdf")
# fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
# plt.fill_between(x_star[:, 0], lower[:, 1], upper[:, 1], 
#                     facecolor='orange', alpha=0.5, label=r"$2\sigma_{T,hf}(x)$")
# plt.plot(x_star, u_star[:, 1])
# plt.plot(x_star, u_pred[:, 1], "r--")
# plt.scatter(x_train, u_train[:, 1],)
# plt.savefig("results/sin.pdf")