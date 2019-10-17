# From https://github.com/maziarraissi/PINNs (MIT Lincese, maziarraissi)
import numpy as np
import os
import sys
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime


def figsize(n_plot_x, n_plot_y, scale=1.):
    plot_width_pt = 100.                          # Get this from LaTeX using \the\textwidth
    plot_height_pt = 100.                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = n_plot_y*plot_width_pt*inches_per_pt*scale    # width in inches
    fig_height = n_plot_x*plot_height_pt*inches_per_pt*scale    # width in inches
    return [fig_width,fig_height]

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
    "figure.figsize": figsize(1, 1),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)


def saveresultdir(save_path, save_hp):
    now = datetime.now()
    scriptname = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    resdir = os.path.join(save_path, "results", f"{now.strftime('%y%m%d-%H%M%S')}-{scriptname}")
    os.mkdir(resdir)
    print("saving results to directory ", resdir)
    with open(os.path.join(resdir, "hp.json"), "w") as f:
        json.dump(save_hp, f)
    filename = os.path.join(resdir, "graph")
    savefig(filename)


def savefig(filename):
    # plt.savefig("{}.pdf".format(filename))
    # plt.savefig("{}.png".format(filename))
    plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0)
    plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
    plt.close()

