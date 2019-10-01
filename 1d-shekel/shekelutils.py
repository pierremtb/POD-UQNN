import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import json


# From https://github.com/maziarraissi/PINNs (MIT Lincese, maziarraissi)
def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

    pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
    plt.rcParams.update(pgf_with_latex)


def saveResultDir(save_path, save_hp):
    now = datetime.now()
    scriptName =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
    resDir = os.path.join(save_path, "results", f"{now.strftime('%Y%m%d-%H%M%S')}-{scriptName}")
    os.mkdir(resDir)
    print("Saving results to directory ", resDir)
    filename = os.path.join(resDir, "graph")
    plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
    plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0)
    with open(os.path.join(resDir, "hp.json"), "w") as f:
        json.dump(save_hp, f)


def plot_results(X_u_rb_test, u_rb_test, u_rb_pred, u_h, u_h_pred, hp, save_path=None):

    x = np.linspace(0, 10, hp["n_e"])
    fig = plt.figure(figsize=figsize(2, 2))

    # Plotting the first three coefficients u_rb
    ax0 = fig.add_subplot(2, 2, 1)
    for i in range(3):
        ax0.plot(np.sort(X_u_rb_test[:, 0]), u_rb_test[:, i][np.argsort(X_u_rb_test[:, 0])], "--")
        ax0.scatter(X_u_rb_test[:, 0], u_rb_pred[:, i], marker="+")
    ax0.set_title(r"$u_{rb}$ coefficients for $\gamma_0$")
    
    # Plotting the first three coefficients u_rb
    ax00 = fig.add_subplot(2, 2, 2)
    for i in range(3):
        ax00.plot(np.sort(X_u_rb_test[:, 1]), u_rb_test[:, i][np.argsort(X_u_rb_test[:, 1])], "--")
        ax00.scatter(X_u_rb_test[:, 1], u_rb_pred[:, i], marker="+")
    ax00.set_title(r"$u_{rb}$ coefficients for $\gamma_0$")
    
    # Plotting the means
    ax1 = fig.add_subplot(2, 2, 3)
    ax1.plot(x, np.mean(u_h_pred, axis=1), "b-", label=r"$\hat{u_h}(x, \mu)$")
    ax1.plot(x, np.mean(u_h, axis=1), "r--", label=r"$u_h(x, \mu)$")
    ax1.legend()
    ax1.set_title("Means")

    ax2 = fig.add_subplot(2, 2, 4)
    ax2.plot(x, np.std(u_h_pred, axis=1), "b-", label=r"$\hat{u_h}(x, \mu)$")
    ax2.plot(x, np.std(u_h, axis=1), "r--", label=r"$u_h(x, \mu)$")
    ax2.legend()
    ax2.set_title("Standard deviations")
    
    if save_path != None:
        saveResultDir(save_path, save_hp=hp)
    else:
        plt.show()
