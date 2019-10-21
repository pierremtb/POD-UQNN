import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from pyDOE import lhs
from deap.benchmarks import shekel
import json

eqnPath = "1d-burgers"
sys.path.append("utils")
from plotting import figsize, saveresultdir, savefig
from metrics import error_podnn
sys.path.append(os.path.join("datagen", eqnPath))
sys.path.append(os.path.join(eqnPath, "burgersutils"))
from names import X_FILE, U_MEAN_FILE, U_STD_FILE
from burgers import burgers_viscous_time_exact1 as burgers_u


def restruct(U, n_x, n_t, n_s):
    return np.reshape(U, (n_x, n_t, n_s))


def prep_data(n_x, x_min, x_max, n_t, t_min, t_max, n_s, mu_mean):
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
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_t)
    tT = t.reshape((n_t, 1))
    for i in tqdm(range(n_s)):
        # Calling the analytical solution function
        U[:, i:i+n_t] = burgers_u(mu_lhs[i, :], n_x, x, n_t, t) 
        X_v[i:i+n_t, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))
    return U, X_v, lb, ub


def get_test_data():
    dirname = os.path.join(eqnPath, "data")
    x = np.load(os.path.join(dirname, X_FILE))
    U_test_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
    U_test_std = np.load(os.path.join(dirname, U_STD_FILE))
    return x, U_test_mean, U_test_std


def plot_inf_cont_results(X_star, U_pred, Sigma_pred, X_u_train, u_train, Exact_u,
  X, T, x, t, save_path=None, save_hp=None):

  # Interpolating the results on the whole (x,t) domain.
  # griddata(points, values, points at which to interpolate, method)
  # U_pred = griddata(X_star, u_pred, (X, T), method='cubic')

  # Creating the figures
  fig, ax = newfig(1.0, 1.1)
  ax.axis('off')

  X_u = X_u_train[:, 0:1]
  T_u = X_u_train[:, 1:2]
  Y_u = u_train
  Exact = Exact_u


  ####### Row 0: u(t,x) ##################    
  gs0 = gridspec.GridSpec(1, 2)
  gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
  ax = plt.subplot(gs0[:, :])
  
  h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(h, cax=cax)
  
  ax.plot(T_u, X_u, 'kx', label = 'Data (%d points)' % (Y_u.shape[0]), markersize = 4, clip_on = False)
  
  line = np.linspace(x.min(), x.max(), 2)[:,None]
  ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
  ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
  ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
  
  ax.set_xlabel('$t$')
  ax.set_ylabel('$x$')
  ax.legend(frameon=False, loc = 'best')
  ax.set_title('$u(t,x)$', fontsize = 10)


  ####### Row 1: u(t,x) slices ##################    
  gs1 = gridspec.GridSpec(1, 3)
  gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
  
  ax = plt.subplot(gs1[0, 0])
  ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
  ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
  lower = U_pred[25,:] - 2.0*np.sqrt(Sigma_pred[25,:])
  upper = U_pred[25,:] + 2.0*np.sqrt(Sigma_pred[25,:])
  plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
  ax.set_xlabel('$x$')
  ax.set_ylabel('$u(t,x)$')    
  ax.set_title('$t = 0.25$', fontsize = 10)
  ax.axis('square')
  ax.set_xlim([-1.1,1.1])
  ax.set_ylim([-1.1,1.1])
  
  ax = plt.subplot(gs1[0, 1])
  ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
  ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
  lower = U_pred[50,:] - 2.0*np.sqrt(Sigma_pred[50,:])
  upper = U_pred[50,:] + 2.0*np.sqrt(Sigma_pred[50,:])
  plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
  ax.set_xlabel('$x$')
  ax.set_ylabel('$u(t,x)$')
  ax.axis('square')
  ax.set_xlim([-1.1,1.1])
  ax.set_ylim([-1.1,1.1])
  ax.set_title('$t = 0.50$', fontsize = 10)
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
  
  ax = plt.subplot(gs1[0, 2])
  ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
  ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
  lower = U_pred[75,:] - 2.0*np.sqrt(Sigma_pred[75,:])
  upper = U_pred[75,:] + 2.0*np.sqrt(Sigma_pred[75,:])
  plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
  ax.set_xlabel('$x$')
  ax.set_ylabel('$u(t,x)$')
  ax.axis('square')
  ax.set_xlim([-1.1,1.1])
  ax.set_ylim([-1.1,1.1])    
  ax.set_title('$t = 0.75$', fontsize = 10)


  # savefig('./Prediction')
  

  # fig, ax = newfig(1.0)
  # ax.axis('off')
  
  # #############       Uncertainty       ##################
  # gs2 = gridspec.GridSpec(1, 2)
  # gs2.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
  # ax = plt.subplot(gs2[:, :])
  
  # h = ax.imshow(Sigma_pred.T, interpolation='nearest', cmap='rainbow', 
  #               extent=[t.min(), t.max(), x.min(), x.max()], 
  #               origin='lower', aspect='auto')
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="5%", pad=0.05)
  # fig.colorbar(h, cax=cax)
  # ax.set_xlabel('$t$')
  # ax.set_ylabel('$x$')
  # ax.legend(frameon=False, loc = 'best')
  # ax.set_title('Variance of $u(t,x)$', fontsize = 10)

  if save_path != None and save_hp != None:
      saveResultDir(save_path, save_hp)

  else:
    plt.show()

def plot_results(U, U_pred=None,
                 hp=None, save_path=None):

    x, U_test_mean, U_test_std = get_test_data()

    U_pred_mean = np.mean(U_pred, axis=1)
    U_pred_std = np.std(U_pred, axis=1)
    error_test_mean = 100 * error_podnn(U_test_mean, U_pred_mean)
    error_test_std = 100 * error_podnn(U_test_std, U_pred_std)
    if save_path is not None:
        print("--")
        print(f"Error on the mean test HiFi LHS solution: {error_test_mean:4f}%")
        print(f"Error on the stdd test HiFi LHS solution: {error_test_std:4f}%")
        print("--")

    fig = plt.figure(figsize=figsize(1, 2, 2))

    # Plotting the means
    ax1 = fig.add_subplot(1, 2, 1)
    if U_pred is not None:
        ax1.plot(x, np.mean(U_pred, axis=1), "b-", label=r"$\hat{u_V}(x)$")
    ax1.plot(x, np.mean(U, axis=1), "r--", label=r"$u_V(x)$")
    ax1.plot(x, U_test_mean, "r,", label=r"$u_T(x)$")
    ax1.legend()
    ax1.set_title("Means")
    ax1.set_xlabel("$x$")

    # Plotting the std
    ax2 = fig.add_subplot(1, 2, 2)
    if U_pred is not None:
        ax2.plot(x, np.std(U_pred, axis=1), "b-", label=r"$\hat{u_V}(x)$")
    ax2.plot(x, np.std(U, axis=1), "r--", label=r"$u_V(x)$")
    ax2.plot(x, U_test_std, "r,", label=r"$u_T(x)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    ax2.set_xlabel("$x$")
    
    if save_path is not None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()

    # Table display of the errors
    # ax = fig.add_subplot(2, 2, 3)
    # ax.axis('off')
    # table = r"\textbf{Numerical results}  \\ \\ " + \
    #         r"\begin{tabular}{|l|c|} " + \
    #         r"\hline " + \
    #         r"Validation error (%.1f\%% of the dataset) & $%.4f \%%$ \\ " % (100 * hp["train_val_ratio"], error_val) + \
    #         r"\hline " + \
    #         r"Test error (HiFi LHS sampling) & $%.4f \%%$ \\ " % (error_test_mean) + \
    #         r"\hline " + \
    #         r"\end{tabular}"
    # ax.text(0.1, 0.1, table)
