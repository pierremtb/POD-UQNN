"""POD-NN modeling for 1D Shekel Equation."""

import sys
import os
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.mesh import create_linear_mesh
from podnn.plotting import genresultdir

from podnn.varneuralnetwork import VarNeuralNetwork
from podnn.metrics import re_mean_std, re
from podnn.mesh import create_linear_mesh
from podnn.logger import Logger
from podnn.advneuralnetwork import NORM_MEANSTD, NORM_NONE


# Datagen
x_star = np.linspace(-6, 6, 100).reshape(100, 1)
u_star = x_star**3
x_val = np.linspace(-4, 4, 100).reshape(100, 1)
u_val = x_star**3
N = 20
lb = int(2/(2*6) * 100)
ub = int((2+2*4)/(2*6) * 100)
idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
noise_std = 6
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

# Model creation
def gen_and_train_model():
    layers = [1, 50, 50, 1]
    epochs = 5000
    lr = 0.001
    model = VarNeuralNetwork(layers, lr, 1e-10, lb=x_train.mean(), ub=x_train.std())
    model.summary()
    logger = Logger(epochs, 100)

    def get_val_err():
        u_val_pred, u_val_pred_var = model.predict(x_val)
        return {
            "RE": re(u_val_pred, u_val),
            "var": u_val_pred_var.mean(),
        }
    logger.set_val_err_fn(get_val_err)

    # Training
    model.fit(x_train, u_train, epochs, logger)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    return model.predict(x_star)

M = 5
u_pred_samples = np.zeros((100, 1, M))
u_pred_var_samples = np.zeros((100, 1, M))
for i in range(0, M):
    u_pred_samples[:, :, i], u_pred_var_samples[:, :, i] = gen_and_train_model()

u_pred = u_pred_samples.mean(-1)
u_pred_var = (u_pred_var_samples + u_pred_samples ** 2).mean(-1) - u_pred ** 2


# u_pred, u_pred_var = model.predict_f(x_star)
# u_pred_samples = model.predict_f_samples(x_star, 10)

plt.plot(x_star, u_star)
plt.scatter(x_train, u_train)
plt.plot(x_star, u_pred, "r--")
lower = u_pred - 2 * np.sqrt(u_pred_var)
upper = u_pred + 2 * np.sqrt(u_pred_var)
plt.fill_between(x_star[:, 0], lower[:, 0], upper[:, 0], 
                    facecolor='orange', alpha=0.5, label=r"$2\sigma_{T,hf}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'orange', linewidth=.5)
plt.show()