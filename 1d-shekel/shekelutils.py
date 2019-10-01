import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import json

sys.path.append("utils")
from plotting import figsize

def saveresultdir(save_path, save_hp):
    now = datetime.now()
    scriptname =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
    resdir = os.path.join(save_path, "results", f"{now.strftime('%y%m%d-%h%m%s')}-{scriptname}")
    os.mkdir(resdir)
    print("saving results to directory ", resdir)
    with open(os.path.join(resdir, "hp.json"), "w") as f:
        json.dump(save_hp, f)
    filename = os.path.join(resdir, "graph")
    savefig(filename)

def savefig(filename):
    plt.savefig('{}.pdf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))
    # plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0)
    # plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_results(X_U_rb_test, U_rb_test, U_rb_pred, U_h, U_h_pred, hp, save_path=None):
    x = np.linspace(0, 10, hp["n_e"])

    fig = plt.figure(figsize=figsize(2, 2))

    # plotting the first three coefficients u_rb
    ax0 = fig.add_subplot(2, 2, 1)
    for i in range(2):
        ax0.plot(np.sort(X_U_rb_test[:, 0]), U_rb_pred[:, i][np.argsort(X_U_rb_test[:, 0])],
                 "b-", label=r"$\hat{u_{rb}}(\gamma_1)$")
        ax0.plot(np.sort(X_U_rb_test[:, 0]), U_rb_test[:, i][np.argsort(X_U_rb_test[:, 0])],
                 "r--", label=r"$u_{rb}(\gamma_1)$")
    ax0.legend() 
    ax0.set_title(r"First two $U_{rb}$ coefficients")
    ax0.set_xlabel(r"$\gamma_1$")

    # plotting the first three coefficients u_rb
    if X_U_rb_test.shape[1] > 1:
        ax00 = fig.add_subplot(2, 2, 2)
        for i in range(2):
            ax00.plot(np.sort(X_U_rb_test[:, 1]), U_rb_pred[:, i][np.argsort(X_U_rb_test[:, 1])],
                     "b-", label=r"$\hat{u_{rb}}(\gamma_2)$")
            ax00.plot(np.sort(X_U_rb_test[:, 1]), U_rb_test[:, i][np.argsort(X_U_rb_test[:, 1])],
                     "r--", label=r"$u_{rb}(\gamma_2)$")
        ax00.legend() 
        ax00.set_title(r"First two $U_{rb}$ coefficients")
        ax00.set_xlabel(r"$\gamma_2$")
        
    # plotting the means
    ax1 = fig.add_subplot(2, 2, 3)
    ax1.plot(x, np.mean(U_h_pred, axis=1), "b-", label=r"$\hat{U_h}(x, \mu)$")
    ax1.plot(x, np.mean(U_h, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax1.legend()
    ax1.set_title("Means")

    ax2 = fig.add_subplot(2, 2, 4)
    ax2.plot(x, np.std(U_h_pred, axis=1), "b-", label=r"$\hat{U_h}(x, \mu)$")
    ax2.plot(x, np.std(U_h, axis=1), "r--", label=r"$U_h(x, \mu)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    
    if save_path != None:
        saveresultdir(save_path, save_hp=hp)
    else:
        plt.show()
