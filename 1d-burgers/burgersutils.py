from metrics import error_podnn
from plotting import figsize, saveresultdir, savefig
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import sys
import os
from tqdm import tqdm
from pyDOE import lhs
from deap.benchmarks import shekel
import json

eqnPath = "1d-burgers"
sys.path.append("utils")
sys.path.append(os.path.join("datagen", eqnPath))
sys.path.append(os.path.join(eqnPath, "burgersutils"))
from burgers import burgers_viscous_time_exact1 as burgers_u
from names import X_FILE, T_FILE, U_MEAN_FILE, U_STD_FILE
from handling import scarcify
from pod import get_pod_bases


def restruct(U, n_x, n_t, n_s):
    return np.reshape(U, (n_x, n_t, n_s))


def prep_data(n_x, x_min, x_max, n_t, t_min, t_max, n_s,
        mu_mean, t_v_ratio, eps):
    # Total number of snapshots
    nn_s = n_t*n_s

    # LHS sampling (first uniform, then perturbated)
    print("Doing the LHS sampling...")
    pbar = tqdm(total=100)
    X = lhs(n_s, 1).T
    pbar.update(50)
    lb = mu_mean * (1 - np.sqrt(3)/10)
    ub = mu_mean * (1 + np.sqrt(3)/10)
    mu_lhs = lb + (ub - lb)*X
    pbar.update(50)
    pbar.close()

    # Number of inputs in time plus number of parameters
    n_d = 1 + 1

    # Creating the snapshots
    print(f"Generating {nn_s} corresponding snapshots")
    U = np.zeros((n_x, nn_s))
    X_v = np.zeros((nn_s, n_d))
    # X_v = np.zeros((n_s, n_t, n_d))
    # U_struct = np.zeros((n_x, n_t, n_s))
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    tT = t.reshape((n_t, 1))
    for i in tqdm(range(n_s)):
        # Calling the analytical solution function
        s = n_t * i
        e = n_t * (i + 1)
        U[:, s:e] = burgers_u(mu_lhs[i, :], n_x, x, n_t, t)
        X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))
        # X_v_struct[i, :, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))
        # U_struct[:, :, i] = burgers_u(mu_lhs[i, :], n_x, x, n_t, t)

    # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
    # u_rb are the reduced coefficients we're looking for
    V = get_pod_bases(U, eps)

    # Projecting
    v = (V.T.dot(U)).T
   
    # Splitting the dataset (X_v, v)
    nn_s_train = int(t_v_ratio * nn_s)
    print(X_v.shape)
    print(v.shape)
    print(nn_s_train)
    X_v_train, v_train = X_v[:nn_s_train, :], v[:nn_s_train, :]
    X_v_val, v_val = X_v[nn_s_train:, :], v[nn_s_train:, :]
    print(X_v_train.shape)
    print(v_train.shape)
    print(X_v_val.shape)
    print(v_val.shape)
    exit(0)
    # X_v_train, v_train, X_v_val, v_val = \
    #         scarcify(X_v, v, n_s_train)
    
    U_val = V.dot(v_val.T)

    return X_v_train, v_train, X_v_val, v_val, \
        lb, ub, V, U_val


def get_test_data():
    dirname = os.path.join(eqnPath, "data")
    X = np.load(os.path.join(dirname, X_FILE))
    T = np.load(os.path.join(dirname, T_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return X, T, U_test_mean, U_test_std


def plot_contour(fig, pos, X, T, U, levels, title):
    ax = fig.add_subplot(pos)
    ct = ax.contourf(X, T, U, origin="lower")
    # ct = ax.contourf(X, T, U, levels=levels, origin="lower")
    plt.colorbar(ct)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")


def plot_map(fig, pos, x, t, X, T, U, title):
    XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    U_test_grid = griddata(XT, U.flatten(), (X, T), method='cubic')
    ax = fig.add_subplot(pos)
    h = ax.imshow(U_test_grid, interpolation='nearest', cmap='rainbow', 
            extent=[t.min(), t.max(), x.min(), x.max()], 
            origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title(title)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")


def plot_spec_time(fig, pos, x, t_i, U_pred, U_val, U_test, title):
    ax = fig.add_subplot(pos)
    ax.plot(x, U_pred[:, t_i], "b-")
    ax.plot(x, U_val[:, t_i], "r--")
    ax.plot(x, U_test[:, t_i], "k,")
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_title(title)


def plot_results(U_val, U_pred,
                 hp=None, save_path=None):
    X, T, U_test_mean, U_test_std = get_test_data()
    t = T[0, :]
    x = X[:, 0]

    U_pred_mean = np.mean(U_pred, axis=2)
    U_pred_std = np.std(U_pred, axis=2)
    U_val_mean = np.mean(U_val, axis=2)
    U_val_std = np.std(U_val, axis=2)
    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    if save_path is not None:
        print("--")
        print(f"Error on the mean test HiFi LHS solution: {error_test_mean:.4f}%")
        print(f"Error on the stdd test HiFi LHS solution: {error_test_std:.4f}%")
        print("--")

    mean_levels = list(range(2, 15))
    std_levels = np.arange(5, 20) * 0.1

    n_plot_x = 5
    n_plot_y = 3
    fig = plt.figure(figsize=figsize(n_plot_x, n_plot_y, scale=2.0))
    gs = fig.add_gridspec(n_plot_x, n_plot_y)

    plot_map(fig, gs[0, :n_plot_y], x, t, X, T, U_pred_mean, "Mean $u(x,t)$ [pred]")
    plot_map(fig, gs[1, :n_plot_y], x, t, X, T, U_val_mean, "Mean $u(x,t)$ [val]")
    plot_map(fig, gs[2, :n_plot_y], x, t, X, T, U_test_mean, "Mean $u(x,t)$ [test]")
    plot_spec_time(fig, gs[3, 0], x, 25, 
            U_pred_mean, U_val_mean, U_test_mean, "Means $u(x, t=0.25)$")
    plot_spec_time(fig, gs[3, 1], x, 50,
            U_pred_mean, U_val_mean, U_test_mean, "Means $u(x, t=0.50)$")
    plot_spec_time(fig, gs[3, 2], x, 75,
            U_pred_mean, U_val_mean, U_test_mean, "Means $u(x, t=0.75)$")
    plot_spec_time(fig, gs[4, 0], x, 25,
            U_pred_std, U_val_std, U_test_std, "Std dev $u(x, t=0.25)$")
    plot_spec_time(fig, gs[4, 1], x, 50,
            U_pred_std, U_val_std, U_test_std, "Std dev $u(x, t=0.50)$")
    plot_spec_time(fig, gs[4, 2], x, 75,
            U_pred_std, U_val_std, U_test_std, "Std dev $u(x, t=0.75)$")

    plt.tight_layout()
    if save_path is not None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()     
