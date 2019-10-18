import sys
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1111)
tf.random.set_seed(1111)

eqnPath = "2d-ackley"
sys.path.append(eqnPath)
sys.path.append("utils")
from pod import get_pod_bases
from neuralnetwork import NeuralNetwork
from logger import Logger
from ackleyutils_cont import plot_results, prep_data, restruct

# HYPER PARAMETERS

if len(sys.argv) > 1:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # DOF per solution point
    hp["n_h"] = 1
    # Space coordinates
    hp["n_x"] = 100
    hp["x_min"] = -5.
    hp["x_max"] = 1.
    hp["n_y"] = 100
    hp["y_min"] = -5.
    hp["y_max"] = 5.
    # Snapshots count
    hp["n_s"] = 100
    # Train/Val repartition
    hp["train_val_ratio"] = 0.5
    # POD stopping param
    hp["eps"] = 1e-10
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 2000
    hp["tf_lr"] = 0.01
    hp["tf_decay"] = 0.
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    hp["lambda"] = 1e-6
    hp["log_frequency"] = 1

n_x = hp["n_x"]
n_y = hp["n_y"]
n_s = hp["n_s"]
x_min = hp["x_min"]
x_max = hp["x_max"]
y_min = hp["y_min"]
y_max = hp["y_max"]

# Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
# u_rb are the reduced coefficients we're looking for
X, Y, U_train, X_v_star, lb, ub = prep_data(hp["n_h"], n_x, n_y, n_s, x_min, x_max, y_min, y_max)
V = get_pod_bases(U_train, hp["eps"])

# Sizes
n_L = V.shape[1]
n_d = X_v_star.shape[1]

# Projecting
v_star = (V.T.dot(U_train)).T

# Splitting data
n_s_train = int(hp["train_val_ratio"] * hp["n_s"] * hp["n_x"])
# X_v_train, v_train, X_v_val, v_val = \
#         scarcify(X_v_star, v_star, n_s_train)
i_end_train = int(hp["train_val_ratio"] * hp["n_s"] * hp["n_x"])
X_v_train = X_v_star[:i_end_train, :]
v_train = v_star[:i_end_train, :]
X_v_val = X_v_star[i_end_train:, :]
v_val = v_star[i_end_train:, :]

# Creating the neural net model, and Logger
# In: (gam_0, gam_1, gam_2)
# Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
hp["layers"] = [n_d, 40, 60, n_L]
logger = Logger(hp)
model = NeuralNetwork(hp, logger, ub, lb)

# Setting the error function
def error():
    v_pred = model.predict(X_v_val)
    return 1/v_pred.shape[0] * tf.reduce_sum(tf.square(v_pred - v_val))
logger.set_error_fn(error)

# Training
model.fit(X_v_train, v_train)

# Predicting the coefficients
v_pred = model.predict(X_v_val)
print(f"Error calculated on n_s_train = {n_s_train} samples" +
      f" ({int(100 * hp['train_val_ratio'])}%)")

# Retrieving the function with the predicted coefficients
U_pred = V.dot(v_pred.T)

# Restructuring
n_s_val = int(n_s * hp["train_val_ratio"])
U_pred_struct = restruct(U_pred, n_x, n_s_val)
U_train_struct = restruct(U_train, n_x, n_s)
import matplotlib.pyplot as plt
x_train = np.linspace(hp["x_min"], hp["x_max"], n_x)
x_pred = x_train
plt.plot(x_train, U_train_struct[:, 0])
plt.plot(x_train, U_train_struct[:, 1])
plt.plot(x_train, U_train_struct[:, 2])
plt.show()
plt.plot(x_pred, U_pred_struct[:, 0])
plt.plot(x_pred, U_pred_struct[:, 1])
plt.plot(x_pred, U_pred_struct[:, 2])
plt.show()
# Plotting and saving the results
plot_results(U_train_struct, U_pred_struct, X_v_val, v_val, v_pred, hp)
# plot_results(U, U_pred, X_v_val, v_val, v_pred, hp, eqnPath)
