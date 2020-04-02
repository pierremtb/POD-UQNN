"""POD-NN modeling for 1D time-dep Burgers Equation."""
#%% Import
import sys
import os
import numpy as np

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.mesh import create_linear_mesh
from poduqnn.logger import Logger, LoggerCallback
from poduqnn.metrics import re_s
from poduqnn.plotting import savefig, figsize
from poduqnn.handling import sample_mu

#%% Prepare
from hyperparams import HP as hp
from hyperparams import u
print(hp)

resdir = "cache"
x_mesh = create_linear_mesh(hp["x_min"], hp["x_max"], hp["n_x"])
np.save(os.path.join(resdir, "x_mesh.npy"), x_mesh)
# x_mesh = np.load(os.path.join(resdir, "x_mesh.npy"))

#%% Init the model
model = PodnnModel(resdir, hp["n_v"], x_mesh, hp["n_t"])

#%% Generate the dataset from the mesh and params
X_v_train, v_train, _, \
    X_v_val, v_val, U_val = model.generate_dataset(u, hp["mu_min"], hp["mu_max"],
                                                    hp["n_s"],
                                                    hp["train_val"],
                                                    eps=hp["eps"], n_L=hp["n_L"],
                                                    t_min=hp["t_min"], t_max=hp["t_max"],
                                                    x_noise=hp["x_noise"])

#%% Train
model.initBNN(hp["h_layers"], hp["lr"], 1/X_v_train.shape[0],
              hp["pi_1"], hp["pi_2"], hp["norm"])
model.train(X_v_train, v_train, X_v_val, v_val, hp["epochs"], freq=hp["log_frequency"])

#%% Generate the dataset from the mesh and params
v_pred, _ = model.predict_v(X_v_val)

err_val = re_s(v_val, v_pred)
print(f"RE_v: {err_val:4f}")

# #%%
# yhat = model.regnn.predict_dist(X_v_val)
# for i in [0, 1]:
#     plt.plot(yhat[i].numpy(), "b-")
#     plt.plot(v_val[i], "r--")
# plt.show()

# #%% Sample the new model to generate a test prediction
# mu_lhs = sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# X_v_tst, U_tst, _, _ = \
#     model.create_snapshots(model.n_d, model.n_h, u, mu_lhs,
#                            t_min=hp["t_min"], t_max=hp["t_max"])
# U_pred, U_pred_sig = model.predict(X_v_tst)
# print(f"RE_tst: {re_s(U_tst, U_pred):4f}")
# U_tst = model.restruct(U_tst)[0]
# U_pred = model.restruct(U_pred)[0]

# #%% Samples graph
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.interpolate import griddata
# n_samples = 3
# mu_lhs_in = sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# mu_lhs_out_min = sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
# mu_lhs_out_max = sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
# mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))

# #%% Contours for demo
# n_plot_x = 2
# n_plot_y = 3
# fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
# gs = fig.add_gridspec(n_plot_x, n_plot_y)
# ax = fig.add_subplot(gs[0, :])
# x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
# t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
# xxT, ttT = np.meshgrid(x, t)
# xx, tt = xxT.T, ttT.T
# XT = np.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))

# U_pred_grid = griddata(XT, U_pred.mean(-1).flatten(), (xx, tt), method='cubic')
# h = ax.imshow(U_pred_grid, interpolation='nearest', cmap='rainbow', 
#                 extent=[t.min(), t.max(), x.min(), x.max()], 
#                 origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title(r"$\hat{u_D}(\bar{s_{\textrm{tst}}})$")

# gs = fig.add_gridspec(n_plot_x, n_plot_y)
# ax = fig.add_subplot(gs[1, :])
# U_tst_grid = griddata(XT, U_tst.mean(-1).flatten(), (xx, tt), method='cubic')
# h = ax.imshow(U_tst_grid, interpolation='nearest', cmap='rainbow', 
#                 extent=[t.min(), t.max(), x.min(), x.max()], 
#                 origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(h, cax=cax)
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")

# plt.tight_layout()
# # plt.show()
# savefig("results/graph-means")

# #%% Slices
# n_samples = 1
# n_plot_x = 2*n_samples
# times = [25, 50, 75]
# times = [2, 5, 8]
# n_plot_y = len(times)
# fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
# gs = fig.add_gridspec(n_plot_x, n_plot_y)
# for j, time in enumerate(times):
#     actual_row = 0
#     for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
#         X_v_samples, _, U_samples, _ = \
#             model.create_snapshots(model.n_d, model.n_h, u, mu_lhs,
#                                 t_min=hp["t_min"], t_max=hp["t_max"])
#         # U_samples = np.reshape(U_samples, (hp["n_x"], hp["n_t"], -1))
#         x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
#         idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
#         for col, idx_i in enumerate(idx):
#             lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
#             st = hp["n_t"] * col
#             en = hp["n_t"] * (col + 1)
#             X_i = X_v_samples[st:en, :]
#             U_pred_i, U_pred_i_sig = model.predict(X_i)
#             U_pred_i = np.reshape(U_pred_i, (hp["n_x"], hp["n_t"], -1))
#             U_pred_i_sig = np.reshape(U_pred_i_sig, (hp["n_x"], hp["n_t"], -1))
#             ax = fig.add_subplot(gs[actual_row, j])
#             ax.plot(x, U_pred_i[:, time, 0], "b-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
#             ax.plot(x, U_samples[:, time, col], "r--", label=r"$u_D(s_{" + lbl + r"})$")
#             lower = U_pred_i[:, time, 0] - 3*U_pred_i_sig[:, time, 0]
#             upper = U_pred_i[:, time, 0] + 3*U_pred_i_sig[:, time, 0]
#             ax.fill_between(x, lower, upper, alpha=0.2, label=r"$3\sigma_D(s_{" + lbl + r"})$")
#             ax.set_xlabel(f"$x\ (t={time})$")
#             actual_row += 1
#             if j == len(times) - 1:
#                 ax.legend()
# plt.tight_layout()
# plt.show()
# # savefig("results/graph-samples")
