"""POD-NN modeling for 1D Shekel Equation."""

import sys
import os
import yaml
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

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

x_star, u_star = pickle.load(os.path.join("cache", "xu_star.pkl"))

# Retrieving the models
models_paths = []
params_path = ""
for root, dirs, files in os.walk("cache"):
    for filename in files:
        if filename.endswith(".h5"):
            models_paths.append(filename)
        if filename.endswith(".pkl"):
            params_path = filename

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
u_pred = np.array(u_pred_samples).mean(-1)
u_pred_var = (np.array(u_pred_var_samples) + np.array(u_pred_samples) ** 2).mean(-1) - u_pred ** 2
lower = u_pred - 3 * np.sqrt(u_pred_var)
upper = u_pred + 3 * np.sqrt(u_pred_var)

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_star[:, 0], lower[:, 0], upper[:, 0], 
                    facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.scatter(x_train, u_train[:, 0], c="r", label=r"$u_T(x)$")
plt.plot(x_star, u_star[:, 0], "r--", label=r"$u_*(x)$")
plt.plot(x_star, u_pred[:, 0], label=r"$\hat{u}_*(x)$")
plt.legend()
plt.xlabel("$x$")
plt.savefig("results/ens.pdf")
