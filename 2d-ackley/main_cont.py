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
    hp["n_t"] = 100
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
n_t = hp["n_t"]
x_min = hp["x_min"]
x_max = hp["x_max"]
y_min = hp["y_min"]
y_max = hp["y_max"]

# Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
# u_rb are the reduced coefficients we're looking for
X, Y, U_h_train, X_U_rb_star, lb, ub = prep_data(hp["n_h"], n_x, n_y, n_t, x_min, x_max, y_min, y_max)
V = get_pod_bases(U_h_train, hp["eps"])

# Sizes
n_L = V.shape[1]
n_d = X_U_rb_star.shape[1]

# Projecting
U_rb_star = (V.T.dot(U_h_train)).T

# Splitting data
n_t_train = int(hp["train_val_ratio"] * hp["n_t"] * hp["n_x"])
# X_U_rb_train, U_rb_train, X_U_rb_val, U_rb_val = \
#         scarcify(X_U_rb_star, U_rb_star, n_t_train)
i_end_train = int(hp["train_val_ratio"] * hp["n_t"] * hp["n_x"])
X_U_rb_train = X_U_rb_star[:i_end_train, :]
U_rb_train = U_rb_star[:i_end_train, :]
X_U_rb_val = X_U_rb_star[i_end_train:, :]
U_rb_val = U_rb_star[i_end_train:, :]

# Creating the neural net model, and Logger
# In: (gam_0, gam_1, gam_2)
# Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
hp["layers"] = [n_d, 40, 60, n_L]
logger = Logger(hp)
model = NeuralNetwork(hp, logger, ub, lb)

# Setting the error function
def error():
    U_rb_pred = model.predict(X_U_rb_val)
    return 1/U_rb_pred.shape[0] * tf.reduce_sum(tf.square(U_rb_pred - U_rb_val))
logger.set_error_fn(error)

# Training
model.fit(X_U_rb_train, U_rb_train)

# Predicting the coefficients
U_rb_pred = model.predict(X_U_rb_val)
print(f"Error calculated on n_t_train = {n_t_train} samples" +
      f" ({int(100 * hp['train_val_ratio'])}%)")

# Retrieving the function with the predicted coefficients
U_h_pred = V.dot(U_rb_pred.T)

# Restructuring
n_t_val = int(n_t * hp["train_val_ratio"])
U_h_pred_struct = restruct(U_h_pred, n_x, n_t_val)
U_h_train_struct = restruct(U_h_train, n_x, n_t)
import matplotlib.pyplot as plt
x_train = np.linspace(hp["x_min"], hp["x_max"], n_x)
x_pred = x_train
plt.plot(x_train, U_h_train_struct[:, 0])
plt.plot(x_train, U_h_train_struct[:, 1])
plt.plot(x_train, U_h_train_struct[:, 2])
plt.show()
plt.plot(x_pred, U_h_pred_struct[:, 0])
plt.plot(x_pred, U_h_pred_struct[:, 1])
plt.plot(x_pred, U_h_pred_struct[:, 2])
plt.show()
# Plotting and saving the results
plot_results(U_h_train_struct, U_h_pred_struct, X_U_rb_val, U_rb_val, U_rb_pred, hp)
# plot_results(U_h, U_h_pred, X_U_rb_val, U_rb_val, U_rb_pred, hp, eqnPath)
