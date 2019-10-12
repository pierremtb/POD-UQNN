import numpy as np
import os

eqnPath = "2d-ackley"

arr_n_s = np.loadtxt(os.path.join(eqnPath, "results", "systematic", "n_s.csv"))
arr_tf_epochs = np.loadtxt(os.path.join(eqnPath, "results", "systematic", "tf_epochs.csv"))
errors_test_mean = np.loadtxt(os.path.join(eqnPath, "results", "systematic", "err_t_mean.csv"))
errors_test_std = np.loadtxt(os.path.join(eqnPath, "results", "systematic", "err_t_std.csv"))

def write_table(f, errors):
    f.write(r"\begin{tabular}{|c||cccccc|} " + "\n")
    f.write(r"\hline " + "\n")
    f.write(r"\diagbox{Epochs}{$N_s$} & " +
            " & ".join([str(int(n_s)) for n_s in arr_n_s]) +
            r" \\ " +  "\n")
    f.write(r"\hline\hline " + "\n")
    for i_tf_e, tf_e in enumerate(arr_tf_epochs):
        f.write(f"{int(tf_e)} " + r"& $" + 
                "$ & $".join(
                    [f"{e:.2f}" + f"\%" for e in errors[:, i_tf_e]]
                ) +
                r"$ \\ " + "\n")
    f.write(r"\hline" + "\n") 
    f.write(r"\end{tabular}" + "\n")

with open(os.path.join(eqnPath, "results", "systematic", "tables.tex"),
          "w+") as f:
    write_table(f, errors_test_mean)
    write_table(f, errors_test_std)
