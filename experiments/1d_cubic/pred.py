"""POD-NN modeling for 1D Shekel Equation."""

import sys
import pickle
import os
import yaml
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)

import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.mesh import create_linear_mesh
from poduqnn.plotting import genresultdir

from poduqnn.varneuralnetwork import VarNeuralNetwork
from poduqnn.metrics import re_mean_std, re_max
from poduqnn.mesh import create_linear_mesh
from poduqnn.logger import Logger
from poduqnn.varneuralnetwork import NORM_MEANSTD, NORM_NONE
from poduqnn.plotting import figsize, savefig

# Loading data
with open(os.path.join("cache", "xu_star.pkl"), "rb") as f:
    x_star, u_star = pickle.load(f)

with open(os.path.join("cache", "xu_train.pkl"), "rb") as f:
    x_train, u_train = pickle.load(f)

# Retrieving the models
models_paths = []
params_path = ""
for root, dirs, files in os.walk("cache"):
    for filename in files:
        if filename.endswith(".h5"):
            models_paths.append(os.path.join("cache", filename))
        if filename.startswith("modelparams") and filename.endswith(".pkl"):
            params_path = os.path.join("cache", filename)

# Predictions
models = []
u_pred_samples = []
u_pred_var_samples = []
for model_path in models_paths:
    model = VarNeuralNetwork.load_from(model_path, params_path)
    u, u_var = model.predict(x_star)
    u_pred_samples.append(u)
    u_pred_var_samples.append(u_var)
    models.append(model)

# Plotting
u_pred = np.array(u_pred_samples).mean(0)
u_pred_var = (np.array(u_pred_var_samples) + np.array(u_pred_samples) ** 2).mean(0) - u_pred ** 2
lower = u_pred - 2 * np.sqrt(u_pred_var)
upper = u_pred + 2 * np.sqrt(u_pred_var)

print(u_pred.shape, u_pred_var.shape)
print(u_star.shape)

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_star[:, 0], lower[:, 0], upper[:, 0], 
                    facecolor='C0', alpha=0.3, label=r"$2\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.scatter(x_train, u_train[:, 0], c="r", label=r"$u_T(x)$")
plt.plot(x_star, u_star[:, 0], "r--", label=r"$u_*(x)$")
plt.plot(x_star, u_pred[:, 0], "b-", label=r"$\hat{u}_*(x)$")
plt.legend()
plt.xlabel("$x$")
savefig(os.path.join("results", "uq-toy-ensnn"))
