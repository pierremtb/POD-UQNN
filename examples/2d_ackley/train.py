"""POD-NN modeling for 2D Shekel Equation."""
#%% Imports
import sys
import os
import tensorflow as tf

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
from podnn.metrics import re_s
from podnn.handling import check_distributed_args

from hyperparams import HP as hp

#%% Prep GPUs
distributed, local_num = check_distributed_args()
print(f"Distributed: {distributed}, Local models: {local_num}")
tf.config.set_soft_device_placement(True)
if distributed:
    import horovod.tensorflow as hvd
    hvd.init()
    gpu_id = hvd.local_rank()
    # tf.config.experimental.set_memory_growth(gpu, True)
    phys_devices = tf.config.experimental.get_visible_devices('GPU')
    tf.config.experimental.set_visible_devices(phys_devices[gpu_id], 'GPU')

#%% Train
with tf.device("/GPU:0"):
    #%% Retrieve model
    model = PodnnModel.load("cache")
    X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

    for i in range(local_num):
        model_id = gpu_id if distributed else i
        model.train_model(model_id, X_v_train, v_train, X_v_val, v_val, hp["epochs"],
                          freq=hp["log_frequency"])
        v_pred, _ = model.regnn[i].predict(X_v_val)
        err_val = re_s(U_val, model.project_to_U(v_pred))
        print(f"RE_v: {err_val:4f}")

        model.save_model(model_id)
exit(0)