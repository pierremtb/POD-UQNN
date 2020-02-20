#%% Import
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfk = tf.keras
tfd = tfp.distributions

sys.path.append(os.path.join("..", ".."))
from podnn.plotting import figsize

#%% Datagen
N_star = 100
x_tst = np.linspace(-6, 6, N_star).reshape(-1, 1)
D = 1
y_tst = x_tst**3
# D = 2
# u1_star = np.cos(x_star)
# u2_star = np.sin(x_star)
# u_star = np.column_stack((u1_star[:, 0], u2_star[:, 0]))

N = 20
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
# idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
idx = np.array([26, 23,  4,  3, 27, 64, 58, 30, 18, 16,  2, 31, 65, 15, 11, 17, 57, 28, 34, 50])
x = np.copy(x_tst[lb + idx])
y = np.copy(y_tst[lb + idx])
# noise_std = 0.01*u_train.std(0)
noise_std = 3
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])

#%% Model creation
# Build model.
tf.keras.backend.set_floatx('float64')
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
# Build model.
model = tf.keras.Sequential([
#   tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable, activation="softplus", kl_weight=1/x.shape[0]),
  tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable, activation="softplus", kl_weight=1/x.shape[0]),
  tfp.layers.DenseVariational(20, posterior_mean_field, prior_trainable, activation="softplus", kl_weight=1/x.shape[0]),
  tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/x.shape[0]),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
])

# Do inference.
negloglik = lambda y, rv_y: -tf.reduce_mean(rv_y.log_prob(y))

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.005), loss=negloglik)
model.fit(x, y, epochs=30000, verbose=False)

yhats = [model(x_tst) for _ in range(100)]
avgm = np.zeros_like(x_tst[..., 0])
plt.scatter(x, y[:, 0], c="r", label=r"$u_T(x)$")
plt.plot(x_tst, y_tst[:, 0], "r--", label=r"$u_*(x)$")
for i, yhat in enumerate(yhats):
  m = np.squeeze(yhat.mean())
  s = np.squeeze(yhat.stddev())
  if i < 15:
    plt.plot(x_tst, m, 'b', label='ensemble means' if i == 0 else None, linewidth=1.)
    plt.plot(x_tst, m + 2 * s, 'C0,', linewidth=0.5, label='ensemble means + 2 ensemble stdev' if i == 0 else None);
    plt.plot(x_tst, m - 2 * s, 'C0,', linewidth=0.5, label='ensemble means - 2 ensemble stdev' if i == 0 else None);
  avgm += m
plt.plot(x_tst, avgm/len(yhats), 'b', label='overall mean', linewidth=4)

#%% Predictions and plotting
# # u_pred, u_pred_var = model.predict(x_tst)
# M = 100
# u_pred_samples = np.zeros((N_star, 1, M))
# u_pred_var_samples = np.zeros((N_star, 1, M))
# for i in range(M):
#     dist = model(x_tst)
#     u_pred_samples[:, :, i] = dist.mean()
#     u_pred_var_samples[:, :, i] = dist.variance()
# u_pred = u_pred_samples.mean(-1)
# u_pred_var = (u_pred_var_samples + u_pred_samples ** 2).mean(-1) - u_pred ** 2
# lower = u_pred - 2 * np.sqrt(u_pred_var)
# upper = u_pred + 2 * np.sqrt(u_pred_var)

# fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
# plt.fill_between(x_tst[:, 0], lower[:, 0], upper[:, 0], 
#                     facecolor='C0', alpha=0.3, label=r"$2\sigma_{T}(x)$")
# # plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
# plt.scatter(x, y[:, 0], c="r", label=r"$u_T(x)$")
# plt.plot(x_tst, y_tst[:, 0], "r--", label=r"$u_*(x)$")
# plt.plot(x_tst, u_pred_samples[:, 0], label=r"$\hat{u}_*(x)$")
# # plt.legend()
# plt.xlabel("$x$")
# plt.show()



