"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import tensorflow as tf
import numpy as np
import meshio

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.handling import check_distributed_args

from hyperparams import HP as hp

#%% Prep GPUs
local_num = hp["n_M"]
distributed = check_distributed_args()
print(f"Distributed: {distributed}, Local models:{local_num}")
tf.config.set_soft_device_placement(True)
if distributed:
    import horovod.tensorflow as hvd
    hvd.init()
    gpu_id = hvd.local_rank()
    local_num = 1
    # tf.config.experimental.set_memory_growth(gpu, True)
    phys_devices = tf.config.experimental.get_visible_devices('GPU')
    tf.config.experimental.set_visible_devices(phys_devices[gpu_id], 'GPU')

#%% Train
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

#%%
for i in range(local_num):
    model_id = gpu_id if distributed else i
    model.train_model(model_id, X_v_train, v_train, X_v_val, v_val, hp["epochs"],
                     freq=hp["log_frequency"], div_max=True)
    model.save_model(model_id)
