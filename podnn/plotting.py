"""Utililities for plotting and saving results."""

import os
import platform
import sys
import subprocess
import yaml
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager


# From https://github.com/maziarraissi/PINNs (MIT License, maziarraissi)
def figsize(n_plot_x, n_plot_y, scale=1.):
    plot_width_pt = 100.                          # Get this from LaTeX using \the\textwidth
    plot_height_pt = 100.                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = n_plot_y*plot_width_pt*inches_per_pt*scale    # width in inches
    fig_height = n_plot_x*plot_height_pt*inches_per_pt*scale    # width in inches
    return [fig_width, fig_height]
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

def genresultdir():
    """Generate the results dir name."""
    now = datetime.now()
    scriptname = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    resdir = os.path.join("results", f"{now.strftime('%y%m%d-%H%M%S')}-{scriptname}")
    os.mkdir(resdir)
    print("Saving results to directory ", resdir)
    return resdir

def saveresultdir(resdir, save_HP, errors, train_res=None):
    """Save plots and hyperparams to a subdirectory of './results/'."""
    with open(os.path.join(resdir, "HP.txt"), "w") as f:
         yaml.dump(save_HP, f)
    with open(os.path.join(resdir, "errors.txt"), "w") as f:
         yaml.dump(errors, f)
    if train_res is not None:
        qty_count = train_res[1].shape[1] - 1
        np.savetxt(os.path.join(resdir, "res.txt"), train_res[1],
                   header=train_res[0], delimiter="\t",
                   fmt="\t".join(["%i"] + ["%1.6f"]*qty_count))
    filename = os.path.join(resdir, "graph")
    savefig(filename)
#    openPdfGraph(filename)

def openPdfGraph(filename):
    """Open filename (without extension) in default PDF viewer."""

    filepath = filename + ".pdf"
    if platform.system() == 'Darwin':
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':
        os.startfile(filepath)
    else:
        subprocess.call(('xdg-open', filepath))


def savefig(filename, tight_box=True):
    """Saves current matplotlib plot in an image and a pdf file."""

    if tight_box:
        plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig("{}.png".format(filename))
        plt.savefig("{}.pdf".format(filename))
    plt.close()
