"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import tensorflow as tf
import numpy as np
import meshio

sys.path.append(os.path.join("..", ".."))
from podnn.podnnmodel import PodnnModel
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
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()


# model.initVNNs(hp["n_M"], hp["h_layers"],
#                 hp["lr"], hp["lambda"], hp["adv_eps"], hp["norm"])

#%%
for i in range(local_num):
    model_id = gpu_id if distributed else i
    model.train_model(model_id, X_v_train, v_train, X_v_val, v_val, hp["epochs"],
                     freq=hp["log_frequency"])
    model.save_model(model_id)


U_pred, U_pred_sig = model.predict(X_v_val)
U_pred, U_pred_sig = model.restruct(U_pred), model.restruct(U_pred_sig)
U_val = model.restruct(U_val)
datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = datadir
from podnn.mesh import read_multi_space_sol_input_mesh
sel = np.loadtxt(os.path.join(datadir, "sel.csv"), skiprows=1, delimiter=",")[:, 0].astype("int64")
x_mesh, connectivity, _, _ = \
        read_multi_space_sol_input_mesh(1, hp["n_t"], hp["d_t"], [1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"], sel)
print("Saving to .vtu")
for s in [0, 1, 2]:
    print(f"Sample is {X_v_val[s*hp['n_t']][1]}")
    for i in range(hp["n_t"]):
        meshio.write_points_cells(os.path.join("cache", f"x_u_val_pred_{s}.{i}.vtu"),
                                x_mesh,
                                [("triangle", connectivity)],
                                point_data={
                                        "eta": U_val[0, :, i, s],
                                        "eta_pred": U_pred[0, :, i, s],
                                        "eta_pred_up": U_pred[0, :, i, s] + 2*U_pred_sig[0, :, i, s],
                                        "eta_pred_lo": U_pred[0, :, i, s] - 2*U_pred_sig[0, :, i, s],
                                })