"""POD-NN modeling for 2D Ackley Equation."""
#%% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.handling import sample_mu
from poduqnn.plotting import figsize, savefig

from hyperparams import HP as hp
from hyperparams import u

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()
v_pred_mean, sig_alea = model.predict_v(X_v_train)
_, sig_alea_val = model.predict_v(X_v_val)
print(sig_alea.mean(), sig_alea.min(), sig_alea.max())
print(sig_alea_val.mean(), sig_alea_val.min(), sig_alea_val.max())

pod_sig_v = np.stack((v_train, v_pred_mean), axis=-1).std(-1).mean(0)
print(pod_sig_v.mean(), pod_sig_v.min(), pod_sig_v.max())

#%% Predict and restruct
U_pred, U_pred_sig = model.predict(X_v_val)

#%% Validation metrics
U_pred, _ = model.predict(X_v_val)
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
mu_lhs = sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
X_v_tst, U_tst, _, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
U_pred, U_pred_sig = model.predict(X_v_tst)
print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

#%% Samples graph
n_samples = 2
mu_lhs_in = sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
mu_lhs_out_min = sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
mu_lhs_out_max = sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))


# Contours for demo
n_plot_x = 2
n_plot_y = 3
fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
gs = fig.add_gridspec(n_plot_x, n_plot_y)
x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
y = np.linspace(hp["y_min"], hp["y_max"], hp["n_y"])
xxT, yyT = np.meshgrid(x, y)
xx, yy = xxT.T, yyT.T

# Slices
for col, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
    X_v_samples, U_samples, _, _ = \
        model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
    U_samples = np.reshape(U_samples, (hp["n_x"], hp["n_y"], -1))
                            
    x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
    idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
    for row, idx_i in enumerate(idx):
        lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
        X_i = X_v_samples[idx_i, :].reshape(1, -1)
        U_pred_i, U_pred_i_sig = model.predict(X_i)
        U_pred_i = np.reshape(U_pred_i, (hp["n_x"], hp["n_y"], -1))
        U_pred_i_sig = np.reshape(U_pred_i_sig, (hp["n_x"], hp["n_y"], -1))

        if row == 0 and col == 0:
            ax = fig.add_subplot(gs[0, 0])
            levels = list(range(2, 15))
            ct = ax.contourf(yy, xx, U_samples[:, :, idx_i].T, levels=levels, origin="lower")
            plt.colorbar(ct)
            ax.set_title(r"$u{\scriptsize\textrm{D}}([" + f"{X_i[0, 0]:.2f}," + f"{X_i[0, 1]:.2f}," + f"{X_i[0, 2]:.2f}] )$")
            ax.axis("equal")
            ax.set_xlabel("$y$")
            ax.set_ylabel("$x$")
            ax.axvline(0., color="w", ls="-.")
            ax = fig.add_subplot(gs[1, 0])
            levels = list(range(2, 15))
            ct = ax.contourf(yy, xx, U_pred_i[:, :, 0].T, levels=levels, origin="lower")
            plt.colorbar(ct)
            ax.axis("equal")
            ax.set_title(r"$\hat{u}{\scriptsize\textrm{D}}([" + f"{X_i[0, 0]:.2f}," + f"{X_i[0, 1]:.2f}," + f"{X_i[0, 2]:.2f}] )$")
            ax.set_xlabel("$y$")
            ax.set_ylabel("$x$")
            ax.axvline(0., color="w", ls="-.")

        ax = fig.add_subplot(gs[row, col+1])
        ax.plot(x, U_pred_i[:, 199, 0], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
        ax.plot(x, U_samples[:, 199, idx_i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
        lower = U_pred_i[:, 199, 0] - 2*U_pred_i_sig[:, 199, 0]
        upper = U_pred_i[:, 199, 0] + 2*U_pred_i_sig[:, 199, 0]
        ax.fill_between(x, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
        ax.set_xlabel("$x\ (y=0)$")
        title_st = r"$s=[" + f"{X_i[0, 0]:.2f}," + f"{X_i[0, 1]:.2f}," + f"{X_i[0, 2]:.2f}] "
        title_st += r"\in \Omega{\footnotesize\textrm{out}}$" if col + 1 == 2 else r"\in \Omega$"
        ax.set_title(title_st)
        if col == 0: 
            ax.legend()
plt.tight_layout()
savefig(os.path.join("results", "podensnn-ackley-graph-meansamples"))

# """POD-NN modeling for 1D Shekel Equation."""
# #%% Imports
# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.interpolate import griddata

# sys.path.append(os.path.join("..", ".."))
# from poduqnn.podnnmodel import PodnnModel
# from poduqnn.metrics import re_s
# from poduqnn.plotting import figsize, savefig
# from poduqnn.handling import sample_mu

# from hyperparams import HP as hp
# from hyperparams import u

# #%% Load models
# model = PodnnModel.load("cache")
# X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

# #%% Predict and restruct
# U_pred, U_pred_sig = model.predict(X_v_val)

# #%% Validation metrics
# U_pred, _ = model.predict(X_v_val)
# err_val = re_s(U_val, U_pred)
# print(f"RE_v: {err_val:4f}")

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
# hp["mu_min_out"] = [0.0005]
# hp["mu_max_out"] = [0.0105]
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
# x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
# t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
# xxT, ttT = np.meshgrid(x, t)
# xx, tt = xxT.T, ttT.T
# XT = np.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))

# # Slices
# n_samples = 1
# times = [25, 75]
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

#             if col == 0 and row == 0:
#                 ax = fig.add_subplot(gs[0, 0])
#                 U_tst_grid = griddata(XT, U_samples[:, :, col].flatten(), (xx, tt), method='cubic')
#                 h = ax.imshow(U_tst_grid, interpolation='nearest', cmap='rainbow', 
#                                 extent=[t.min(), t.max(), x.min(), x.max()], 
#                                 origin='lower', aspect='auto')
#                 divider = make_axes_locatable(ax)
#                 cax = divider.append_axes("right", size="5%", pad=0.05)
#                 fig.colorbar(h, cax=cax)
#                 ax.set_xlabel("$t$")
#                 ax.set_ylabel("$x$")
#                 ax.axvline(X_i[times[0], 0], color="w", ls="-.")
#                 ax.set_title(r"$u{\scriptsize\textrm{D}}(s=" + f"{X_i[0, 1]:.4f}" + ")$")

#                 ax = fig.add_subplot(gs[1, 0])
#                 U_pred_grid = griddata(XT, U_pred_i[:, :, 0].flatten(), (xx, tt), method='cubic')
#                 h = ax.imshow(U_pred_grid, interpolation='nearest', cmap='rainbow', 
#                                 extent=[t.min(), t.max(), x.min(), x.max()], 
#                                 origin='lower', aspect='auto')
#                 divider = make_axes_locatable(ax)
#                 cax = divider.append_axes("right", size="5%", pad=0.05)
#                 fig.colorbar(h, cax=cax)
#                 ax.set_xlabel("$t$")
#                 ax.set_ylabel("$x$")
#                 ax.axvline(X_i[times[1], 0], color="w", ls="-.")
#                 ax.set_title(r"$\hat{u}{\scriptsize\textrm{D}}(s=" + f"{X_i[0, 1]:.4f}" + ")$")

#             ax = fig.add_subplot(gs[j, actual_row + 1])
#             ax.plot(x, U_pred_i[:, time, 0], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
#             ax.plot(x, U_samples[:, time, col], "r--", label=r"$u_D(s_{" + lbl + r"})$")
#             lower = U_pred_i[:, time, 0] - 2*U_pred_i_sig[:, time, 0]
#             upper = U_pred_i[:, time, 0] + 2*U_pred_i_sig[:, time, 0]
#             ax.fill_between(x, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
#             ax.set_xlabel(f"$x$")
#             if row == 0:
#                 ax.set_title(r"$s=" + f"{X_i[0, 1]:.4f}" + r" \in \Omega,\ "
#                              + f"t={X_i[time, 0]:.2f}$")
#             else:
#                 ax.set_title(r"$s=" + f"{X_i[0, 1]:.4f}" + r" \in \Omega{\footnotesize\textrm{out}},\ "
#                              + f"t={X_i[time, 0]:.2f}$")
#             actual_row += 1
#             if j == 0 and actual_row == 0:
#                 ax.legend()
# plt.tight_layout()
# savefig("results/podensnn-burger-graph-meansamples")

# """POD-NN modeling for 2D Ackley Equation."""
# #%% Imports
# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# sys.path.append(os.path.join("..", ".."))
# from poduqnn.podnnmodel import PodnnModel
# from poduqnn.metrics import re_s
# from poduqnn.handling import sample_mu
# from poduqnn.plotting import figsize, savefig

# from hyperparams import HP as hp
# from hyperparams import u

# #%% Load models
# model = PodnnModel.load("cache")
# X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()
# v_pred_mean, sig_alea = model.predict_v(X_v_train)
# _, sig_alea_val = model.predict_v(X_v_val)
# print(sig_alea.mean(), sig_alea.min(), sig_alea.max())
# print(sig_alea_val.mean(), sig_alea_val.min(), sig_alea_val.max())

# pod_sig_v = np.stack((v_train, v_pred_mean), axis=-1).std(-1).mean(0)
# print(pod_sig_v.mean(), pod_sig_v.min(), pod_sig_v.max())

# #%% Predict and restruct
# U_pred, U_pred_sig = model.predict(X_v_val)

# #%% Validation metrics
# U_pred, _ = model.predict(X_v_val)
# err_val = re_s(U_val, U_pred)
# print(f"RE_v: {err_val:4f}")

# #%% Sample the new model to generate a test prediction
# mu_lhs = sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# X_v_tst, U_tst, _, _ = \
#     model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
# U_pred, U_pred_sig = model.predict(X_v_tst)
# print(f"RE_tst: {re_s(U_tst, U_pred):4f}")

# #%% Samples graph
# n_samples = 2
# mu_lhs_in = sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# mu_lhs_out_min = sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
# mu_lhs_out_max = sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
# mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))


# # Contours for demo
# n_plot_x = 2
# n_plot_y = 3
# fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
# gs = fig.add_gridspec(n_plot_x, n_plot_y)
# x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
# y = np.linspace(hp["y_min"], hp["y_max"], hp["n_y"])
# xxT, yyT = np.meshgrid(x, y)
# xx, yy = xxT.T, yyT.T
# ax = fig.add_subplot(gs[0, 0])
# U_tst = np.reshape(U_tst, (hp["n_x"], hp["n_y"], -1))
# levels = list(range(2, 15))
# ct = ax.contourf(xx, yy, U_tst.mean(-1), levels=levels, origin="lower")
# plt.colorbar(ct)
# ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
# ax.axis("equal")
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# ax = fig.add_subplot(gs[1, 0])
# U_pred = np.reshape(U_pred, (hp["n_x"], hp["n_y"], -1))
# levels = list(range(2, 15))
# ct = ax.contourf(xx, yy, U_pred.mean(-1), levels=levels, origin="lower")
# plt.colorbar(ct)
# ax.axis("equal")
# ax.set_title(r"$\hat{u_D}(\bar{s_{\textrm{tst}}})$")
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")

# # Slices
# for col, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
#     X_v_samples, U_samples, _, _ = \
#         model.create_snapshots(model.n_d, model.n_h, u, mu_lhs)
#     U_samples = np.reshape(U_samples, (hp["n_x"], hp["n_y"], -1))
                            
#     x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
#     idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
#     for row, idx_i in enumerate(idx):
#         lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
#         X_i = X_v_samples[idx_i, :].reshape(1, -1)
#         U_pred_i, U_pred_i_sig = model.predict(X_i)
#         U_pred_i = np.reshape(U_pred_i, (hp["n_x"], hp["n_y"], -1))
#         U_pred_i_sig = np.reshape(U_pred_i_sig, (hp["n_x"], hp["n_y"], -1))
#         ax = fig.add_subplot(gs[row, col+1])
#         ax.plot(x, U_pred_i[:, 199, 0], "C0-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
#         ax.plot(x, U_samples[:, 199, idx_i], "r--", label=r"$u_D(s_{" + lbl + r"})$")
#         lower = U_pred_i[:, 199, 0] - 2*U_pred_i_sig[:, 199, 0]
#         upper = U_pred_i[:, 199, 0] + 2*U_pred_i_sig[:, 199, 0]
#         ax.fill_between(x, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
#         ax.set_xlabel("$x\ (y=0)$")
#         title_st = r"$s=[" + f"{X_i[0, 0]:.2f}," + f"{X_i[0, 1]:.2f}," + f"{X_i[0, 2]:.2f}] "
#         title_st += r"\in \Omega{\footnotesize\textrm{out}}$" if col + 1 == 2 else r"\in \Omega$"
#         ax.set_title(title_st)
#         if col == len(idx) - 1 and row == 0:
#             ax.legend()
# plt.tight_layout()
# # plt.show()
# savefig("results/podensnn-ackley-graph-meansamples")
