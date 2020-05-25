"""POD-NN modeling for 1D Shekel Equation."""
#%% Imports
import sys
import os
import numpy as np
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

sys.path.append(os.path.join("..", ".."))
from poduqnn.podnnmodel import PodnnModel
from poduqnn.metrics import re_s
from poduqnn.plotting import figsize, savefig
from poduqnn.handling import sample_mu

from hyperparams import HP as hp
from hyperparams import u

#%% Load models
model = PodnnModel.load("cache")
X_v_train, v_train, U_train, X_v_val, v_val, U_val = model.load_train_data()

#%% Predict and restruct
U_pred, U_pred_sig = model.predict(X_v_val)

#%% Validation metrics
U_pred, _ = model.predict(X_v_val)
err_val = re_s(U_val, U_pred)
print(f"RE_v: {err_val:4f}")

#%% Sample the new model to generate a test prediction
mu_lhs = sample_mu(hp["n_s_tst"], np.array(hp["mu_min"]), np.array(hp["mu_max"]))
X_v_tst, U_tst, _, _ = \
    model.create_snapshots(model.n_d, model.n_h, u, mu_lhs,
                           t_min=hp["t_min"], t_max=hp["t_max"])
U_pred, U_pred_sig = model.predict(X_v_tst)
print(f"RE_tst: {re_s(U_tst, U_pred):4f}")
U_tst = model.restruct(U_tst)[0]
U_pred = model.restruct(U_pred, n_t=hp["n_t"] - 1)[0]

#%% Samples graph
# hp["mu_min_out"] = [0.0005]
# hp["mu_max_out"] = [0.0105]
# n_samples = 3
# mu_lhs_in = sample_mu(n_samples, np.array(hp["mu_min"]), np.array(hp["mu_max"]))
# mu_lhs_out_min = sample_mu(n_samples, np.array(hp["mu_min_out"]), np.array(hp["mu_min"]))
# mu_lhs_out_max = sample_mu(n_samples, np.array(hp["mu_max"]), np.array(hp["mu_max_out"]))
# mu_lhs_out = np.vstack((mu_lhs_out_min, mu_lhs_out_max))
mu_lhs_in = np.array([12]).reshape(-1, 1)
mu_lhs_out = np.array([25]).reshape(-1, 1)

X_v_samples, _, _, _ = \
            model.create_snapshots(model.n_d, model.n_h, u, mu_lhs_in,
           t_min=hp["t_min"], t_max=hp["t_max"])
X_v_samples_out, _, _, _ = \
            model.create_snapshots(model.n_d, model.n_h, u, mu_lhs_out,
                                   t_min=hp["t_min"], t_max=hp["t_max"])
U_pred, U_pred_sig = model.predict(X_v_samples)
U_pred_out, U_pred_sig_out = model.predict(X_v_samples_out)

mpiw_tst = 4 * U_pred_sig.mean()
print(f"MPIW_tst: {mpiw_tst:.4e}")
mpiw_tst_out = 4 * U_pred_sig_out.mean()
print(f"MPIW_tst_out: {mpiw_tst_out:.4e}")

#%% Contours for demo
x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
t = np.linspace(hp["t_min"], hp["t_max"], hp["n_t"])
xxT, ttT = np.meshgrid(x, t)
xx, tt = xxT.T, ttT.T
XT = np.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))

# Slices
n_samples = 1
times = [0, 25]
ylim = [[(-3, 28), (-3, 26)], [(-3, 28), (-3, 28)]]

has_sim_data = False
if os.path.exists(os.path.join("data", "sel.csv")):
    has_sim_data = True
    sel = np.loadtxt(os.path.join("data", "sel.csv"), skiprows=1, delimiter=",")[:, 6].astype("int")

for j, time in enumerate(times):
    n_plot_x = 1
    n_plot_y = 3
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)
    actual_row = 0
    for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
        X_v_samples, U_samples, _, _ = \
            model.create_snapshots(model.n_d, model.n_h, u, mu_lhs,
                                t_min=hp["t_min"], t_max=hp["t_max"])
        U_samples = np.reshape(U_samples, (hp["n_v"], hp["n_x"], hp["n_t"], -1))
        U_samples = U_samples[0]
        x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
        idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
        for col, idx_i in enumerate(idx):
            lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
            st = hp["n_t"] * col
            en = hp["n_t"] * (col + 1)
            X_i = X_v_samples[st:en, :]
            if j > 0:
                U_pred_i, U_pred_i_sig = model.predict(X_i)
                U_pred_i = np.reshape(U_pred_i, (hp["n_v"], hp["n_x"], hp["n_t"], -1))
                U_pred_i_sig = np.reshape(U_pred_i_sig, (hp["n_v"], hp["n_x"], hp["n_t"], -1))
                U_pred_i = U_pred_i[0]
                U_pred_i_sig = U_pred_i_sig[0]
            else:
                U_pred_i = U_samples
                U_pred_i_sig = np.zeros_like(U_samples)

            if row == 0 and j == 1:
                ax = fig.add_subplot(gs[0, actual_row])
                U_grid = griddata(XT, U_pred_i.flatten(), (xx, tt), method='cubic')
                h = ax.imshow(U_grid.T, interpolation='nearest', cmap='rainbow', 
                                extent=[x.min(), x.max(), t.min(), t.max()], 
                                origin='lower', aspect='auto')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ax.axhline(X_i[25, 0], color="w", ls="-.")
                fig.colorbar(h, cax=cax)
                ax.set_xlabel(r"$x\ [\textrm{m}]$")
                ax.set_ylabel(r"$t\ [\textrm{s}]$")
                # ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
                ax.set_title(r"$\hat{u}^\mu_D(s=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega)$")

            if row == 0 and j == 0:
                ax = fig.add_subplot(gs[0, actual_row])
                U_grid = griddata(XT, U_samples[:, :, col].flatten(), (xx, tt), method='cubic')
                h = ax.imshow(U_grid.T, interpolation='nearest', cmap='rainbow', 
                                extent=[x.min(), x.max(), t.min(), t.max()], 
                                origin='lower', aspect='auto')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ax.axhline(0.02, color="w", ls="-.")
                fig.colorbar(h, cax=cax)
                ax.set_xlabel(r"$x\ [\textrm{m}]$")
                ax.set_ylabel(r"$t\ [\textrm{s}]$")
                # ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
                ax.set_title(r"$u_D(s=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega)$")

            ax = fig.add_subplot(gs[0, actual_row + 1])

            if has_sim_data:
                vtkfilename = os.path.join("data", f"cas1_{int(X_i[0, 1])}m", f"0_FV-Paraview_{time}.vtk")
                if os.path.exists(vtkfilename):
                    vtk = meshio.read(vtkfilename)
                    x_sim = vtk.points[sel, 1]
                    h_sim = vtk.point_data["h"][sel]
                    ax.plot(x_sim, h_sim, "k:", label=r"$u_\textrm{sim}(s_{" + lbl + r"})$")

            ax.plot(x, U_pred_i[:, time, 0], "b-", label=r"$\hat{u}^\mu_D(s_{" + lbl + r"})$")
            ax.plot(x, U_samples[:, time, col], "r--", label=r"$u_D(s_{" + lbl + r"})$")
            lower = U_pred_i[:, time, 0] - 2*U_pred_i_sig[:, time, 0]
            upper = U_pred_i[:, time, 0] + 2*U_pred_i_sig[:, time, 0]
            ax.fill_between(x, lower, upper, alpha=0.2, label=r"$\pm2\hat{u}^\sigma_D(s_{" + lbl + r"})$")
            if row == 0 and j == 1:
                ax.text(0, 20,  "POD-EnsNN")
            ax.set_xlabel(r"$x\ [\textrm{m}]$")
            ax.set_ylabel(r"$h\ [\textrm{m}]$")
            ax.set_ylim(ylim[j][row])
            if row == 0:
                ax.set_title(r"$\Delta\eta=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega"
                             + f",\ t={X_i[time, 0]:.2f}" + r"\ \textrm{s}$")
            else:
                ax.set_title(r"$\Delta\eta=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega{\footnotesize\textrm{out}}"
                             + f",\ t={X_i[time, 0]:.2f}" + r"\ \textrm{s}$")
            actual_row += 1
            if j == 0:
                ax.legend()
    plt.tight_layout()
    savefig(os.path.join("results", f"podensnn-1dswt-graph-meansamples-h-{j}"), True)

#%% Velocity plots
times = [0, 25]
# ylim = [(-1, 1), (-1, 10)]
ylim = [[(-3, 17), (-3, 17)], [(-3, 17), (-3, 17)]]
for j, time in enumerate(times):
    n_plot_x = 1
    n_plot_y = 3
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)
    actual_row = 0
    for row, mu_lhs in enumerate([mu_lhs_in, mu_lhs_out]):
        X_v_samples, U_samples, _, _ = \
            model.create_snapshots(model.n_d, model.n_h, u, mu_lhs,
                                t_min=hp["t_min"], t_max=hp["t_max"])
        U_samples = np.reshape(U_samples, (hp["n_v"], hp["n_x"], hp["n_t"], -1))
        U_samples = U_samples[1]
        x = np.linspace(hp["x_min"], hp["x_max"], hp["n_x"])
        idx = np.random.choice(X_v_samples.shape[0], n_samples, replace=False)
        for col, idx_i in enumerate(idx):
            lbl = r"{\scriptscriptstyle\textrm{tst}}" if row == 0 else r"{\scriptscriptstyle\textrm{out}}"
            st = hp["n_t"] * col
            en = hp["n_t"] * (col + 1)
            X_i = X_v_samples[st:en, :]
            if j > 0:
                U_pred_i, U_pred_i_sig = model.predict(X_i)
                U_pred_i = np.reshape(U_pred_i, (hp["n_v"], hp["n_x"], hp["n_t"], -1))
                U_pred_i_sig = np.reshape(U_pred_i_sig, (hp["n_v"], hp["n_x"], hp["n_t"], -1))
                U_pred_i = U_pred_i[1]
                U_pred_i_sig = U_pred_i_sig[1]
            else:
                U_pred_i = U_samples
                U_pred_i_sig = np.zeros_like(U_samples)

            if row == 0 and j == 1:
                ax = fig.add_subplot(gs[0, actual_row])
                U_grid = griddata(XT, U_pred_i.flatten(), (xx, tt), method='cubic')
                h = ax.imshow(U_grid.T, interpolation='nearest', cmap='rainbow', 
                                extent=[x.min(), x.max(), t.min(), t.max()], 
                                origin='lower', aspect='auto')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ax.axhline(X_i[25, 0], color="w", ls="-.")
                fig.colorbar(h, cax=cax)
                ax.set_xlabel(r"$x\ [\textrm{m}]$")
                ax.set_ylabel(r"$t\ [\textrm{s}]$")
                # ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
                ax.set_title(r"$\hat{u}_D^\mu(s=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega)$")

            if row == 0 and j == 0:
                ax = fig.add_subplot(gs[0, actual_row])
                U_grid = griddata(XT, U_samples[:, :, col].flatten(), (xx, tt), method='cubic')
                h = ax.imshow(U_grid.T, interpolation='nearest', cmap='rainbow', 
                                extent=[x.min(), x.max(), t.min(), t.max()], 
                                origin='lower', aspect='auto')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ax.axhline(0.02, color="w", ls="-.")
                fig.colorbar(h, cax=cax)
                ax.set_xlabel(r"$x\ [\textrm{m}]$")
                ax.set_ylabel(r"$t\ [\textrm{s}]$")
                # ax.set_title(r"$u_D(\bar{s_{\textrm{tst}}})$")
                ax.set_title(r"$u_D(s=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega)$")

            ax = fig.add_subplot(gs[0, actual_row + 1])

            if has_sim_data:
                vtkfilename = os.path.join("data", f"cas1_{int(X_i[0, 1])}m", f"0_FV-Paraview_{time}.vtk")
                if os.path.exists(vtkfilename):
                    vtk = meshio.read(vtkfilename)
                    x_sim = vtk.points[sel, 1]
                    u_sim = vtk.point_data["velocity"][sel, 1]
                    ax.plot(x_sim, u_sim, "k:", label=r"$u_\textrm{sim}(s_{" + lbl + r"})$")

            ax.plot(x, U_pred_i[:, time, 0], "b-", label=r"$\hat{u}_D(s_{" + lbl + r"})$")
            ax.plot(x, U_samples[:, time, col], "r--", label=r"$u_D(s_{" + lbl + r"})$")
            lower = U_pred_i[:, time, 0] - 2*U_pred_i_sig[:, time, 0]
            upper = U_pred_i[:, time, 0] + 2*U_pred_i_sig[:, time, 0]
            ax.fill_between(x, lower, upper, alpha=0.2, label=r"$2\sigma_D(s_{" + lbl + r"})$")
            if row == 0 and j == 1:
                ax.text(0, 10, "POD-EnsNN")
            ax.set_xlabel(r"$x\ [\textrm{m}]$")
            ax.set_ylabel(r"$u\ [\textrm{m/s}]$")
            ax.set_ylim(ylim[j][row])
            if row == 0:
                ax.set_title(r"$\Delta\eta=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega"
                             + f",\ t={X_i[time, 0]:.2f}" + r"\ \textrm{s}$")
            else:
                ax.set_title(r"$\Delta\eta=" + f"{X_i[0, 1]:.1f}" + r"\textrm{ m}\in \Omega{\footnotesize\textrm{out}}"
                             + f",\ t={X_i[time, 0]:.2f}" + r"\ \textrm{s}$")
            actual_row += 1
            if j == 0:
                ax.legend()
    plt.tight_layout()
    savefig(os.path.join("results", f"podensnn-1dswt-graph-meansamples-u-{j}"), True)

