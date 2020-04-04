"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import pickle
import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.ndimage

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.mesh import read_multi_space_sol_input_mesh
from poduqnn.plotting import savefig, figsize

from hyperparams import HP as hp

#%% Load models
model = PodnnModel.load("cache")
# X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()
# X_v_train_0, v_train_0, U_train_0, X_v_val_0, v_val_0, U_val_0 = model.load_init_data()

# #%% Predict and restruct
# U_pred, U_pred_sig = model.predict(X_v_val)

# #%% Validation metrics
# U_pred, _ = model.predict(X_v_val)
# err_val = re_s(U_val, U_pred, div_max=True)
# print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
with open(os.path.join("cache", "train_tst_idx.pkl"), "rb") as f:
        train_tst_idx = pickle.load(f)
# datadir = os.path.join("..", "..", "..", "scratch", "multi2swt") 
datadir = "data"
mu_path = os.path.join(datadir, "INPUT_MONTE_CARLO.dat")
x_u_mesh_path = datadir
sel = np.loadtxt(os.path.join(datadir, "sel.csv"), skiprows=1, delimiter=",")[:, 0].astype("int64")
x_mesh, connectivity, X_v_tst, U_tst = \
        read_multi_space_sol_input_mesh(hp["n_s_tst"], hp["n_t"], hp["d_t"], train_tst_idx[1],
                                        hp["mesh_idx"],
                                        x_u_mesh_path, mu_path,
                                        hp["mu_idx"], sel)
print("Elements count: ", connectivity.shape[0])
print("Nodes count: ", x_mesh.shape[0])
U_pred, U_pred_sig = model.predict(X_v_tst)
U_tst_des = model.destruct(U_tst)
err_val = re_s(U_tst_des, U_pred, div_max=True)
print(f"RE_tst: {err_val:4f}")

U_pred = model.restruct(U_pred)
U_pred_sig = model.restruct(U_pred_sig)

U_pred_0 = model.project_to_U(model.project_to_v(U_tst_des))
U_pred_0 = model.restruct(U_pred_0)

#%% Interp and plot
x = x_mesh[:, 0]
y = x_mesh[:, 1]
# X, Y = np.mgrid[int(x.min()):int(x.max()), int(y.min()):int(y.max())]
xint = np.linspace(x.min(), x.max(), 2000)
yint = np.linspace(y.min(), y.max(), 2000)
X, Y = np.meshgrid(xint, yint)
# X, Y = np.mgrid[0:1000, 0:1000]

dist_pts = [([277182.62, 277179.72], [5048838.87, 5048840.58]),
            ([277206.94, 277211.94], [5048835.51, 5048831.72])]

method = "linear"

for s in [0]:
    for t in range(1, hp["n_t"] - 1):
        h = U_tst[0, :, t, s]
        h_pred = U_pred[0, :, t, s]
        h_pred_up = U_pred[0, :, t, s] + 2 * U_pred_sig[0, :, t, s]
        h_pred_lo = U_pred[0, :, t, s] - 2 * U_pred_sig[0, :, t, s]

        H_pred = griddata((x, y), h_pred, (X, Y), method=method)
        H_pred_up = griddata((x, y), h_pred_up, (X, Y), method=method)
        H_pred_lo = griddata((x, y), h_pred_lo, (X, Y), method=method)
        H = griddata((x, y), h, (X, Y), method=method)
        plt.imshow(H_pred)
        # plt.contourf(X, Y, H_pred)

        # Convert the line to pixel/index coordinates
        x_world = np.array([0, 1000])
        y_world = np.array([1000, 1000])
        col = H_pred.shape[1] * (x_world - X.min()) / X.ptp()
        row = H_pred.shape[0] * (y_world - Y.min()) / Y.ptp()

        # Interpolate the line at "num" points...
        num = 1000
        row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

        # Extract the values along the line, using cubic interpolation
        zi = scipy.ndimage.map_coordinates(H_pred, np.vstack((row, col)))
        plt.plot(x_world, y_world, 'ro-')
        savefig("test")
        plt.plot(zi)
        savefig("test2")
        

exit(0)
# %% VTU export
print("Saving to .vtu")
for s in [0]:
# for s in [0, 1, 2]:
    print(f"Sample is {30.0 - X_v_tst[s*hp['n_t']][1]}")
    meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_{s}.{0}.vtu"),
                              x_mesh,
                              [("triangle", connectivity)],
                              point_data={
                                      "eta": U_tst[0, :, 0, s],
                                      "eta_pred": U_pred_0[0, :, 0, s],
                                      "eta_pred_up": U_pred_0[0, :, 0, s],
                                      "eta_pred_lo": U_pred_0[0, :, 0, s],
                              })
    for i in range(1, hp["n_t"] - 1):
        meshio.write_points_cells(os.path.join("cache", f"x_u_tst_pred_{s}.{i}.vtu"),
                                x_mesh,
                                [("triangle", connectivity)],
                                point_data={
                                        "eta": U_tst[0, :, i, s],
                                        "eta_pred": U_pred[0, :, i, s],
                                        "eta_pred_up": U_pred[0, :, i, s] + 2*U_pred_sig[0, :, i, s],
                                        "eta_pred_lo": U_pred[0, :, i, s] - 2*U_pred_sig[0, :, i, s],
                                })
     
