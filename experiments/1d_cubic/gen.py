import sys
import os
import pickle
import yaml
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)

import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from lib.podnnmodel import PodnnModel
from lib.mesh import create_linear_mesh
from lib.plotting import genresultdir

from lib.varneuralnetwork import VarNeuralNetwork
from lib.metrics import re_mean_std, re_max
from lib.mesh import create_linear_mesh
from lib.logger import Logger
from lib.varneuralnetwork import NORM_MEANSTD, NORM_NONE
from lib.plotting import figsize

# Datagen
N_star = 100
D = 1
x_star = np.linspace(-6, 6, N_star).reshape((N_star, 1))
u_star = x_star**3

# Train split
N = 20
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
noise_std = 3
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

# Saving data
with open(os.path.join("cache", "xu_star.pkl"), "wb") as f:
    pickle.dump((x_star, u_star), f)
with open(os.path.join("cache", "xu_train.pkl"), "wb") as f:
    pickle.dump((x_train, u_train), f)
