"""POD-NN modeling for 1D Shekel Equation."""

#%% Imports
import sys
import os
import pickle
import time
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
from podnn.metrics import re_mean_std, re_max
from podnn.mesh import create_linear_mesh
from podnn.logger import Logger
from podnn.varneuralnetwork import NORM_MEANSTD, NORM_NONE
from podnn.plotting import figsize
from podnn.handling import check_distributed_args

distributed, local_num = check_distributed_args()
print(f"Distributed: {distributed}, Local models: {local_num}")

#%% Loading data
with open(os.path.join("cache", "xu_star.pkl"), "rb") as f:
    x_star, u_star = pickle.load(f)
with open(os.path.join("cache", "xu_train.pkl"), "rb") as f:
    x_train, u_train = pickle.load(f)

#%% Prep GPUs
tf.config.set_soft_device_placement(True)
if distributed:
    import horovod.tensorflow as hvd
    hvd.init()
    gpu_id = hvd.local_rank()
    # tf.config.experimental.set_memory_growth(gpu, True)
    phys_devices = tf.config.experimental.get_visible_devices('GPU')
    tf.config.experimental.set_visible_devices(phys_devices[gpu_id], 'GPU')

#%% Hyperparams
layers = [1, 50, 50, 1]
epochs = 20000
lr = 0.0001
dtype = "float64"
tf.keras.backend.set_floatx(dtype)

#%% Training
for i in range(local_num):
    with tf.device("/GPU:0"):
        X = tf.convert_to_tensor(x_train, dtype="float64")
        y = tf.convert_to_tensor(u_train, dtype="float64")

        model = VarNeuralNetwork(layers, lr, 1e-10)
        logger = Logger(epochs, 5000)
        logger.set_val_err_fn(lambda: {})
        model.fit(X, y, epochs, logger)

    # Saving
    gpu_id = time.time()
    model_path = os.path.join("cache", f"model-{gpu_id}.h5")
    params_path = os.path.join("cache", f"modelparams-{gpu_id}.pkl")
    print(f"Saving {model_path} {params_path}")
    model.save_to(model_path, params_path)
