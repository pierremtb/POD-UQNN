import sys
import os
import pickle
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm

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
from poduqnn.plotting import figsize

# Datagen
N_star = 300
D = 1
x_star = np.linspace(-6, 6, N_star).reshape((N_star, 1))
u_star = x_star**3

# Train split
N = 20
lb = int(2/(2*6) * N_star)
ub = int((2+2*4)/(2*6) * N_star)
# idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        91, 100, 188,   8,  63,  98,  78])
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
noise_std = 9
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

# Saving data
with open(os.path.join("cache", "xu_star.pkl"), "wb") as f:
    pickle.dump((x_star, u_star), f)
with open(os.path.join("cache", "xu_train.pkl"), "wb") as f:
    pickle.dump((x_train, u_train), f)
