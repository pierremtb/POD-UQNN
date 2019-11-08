import numpy as np
import os

EQN_PATH = "1d_shekel"


arr_n_s = np.loadtxt(os.path.join(EQN_PATH, "results", "systematic", "n_s.csv"))
arr_tf_epochs = np.loadtxt(os.path.join(EQN_PATH, "results", "systematic", "tf_epochs.csv"))
errors_test_mean = np.loadtxt(os.path.join(EQN_PATH, "results", "systematic", "err_t_mean.csv"))
errors_test_std = np.loadtxt(os.path.join(EQN_PATH, "results", "systematic", "err_t_std.csv"))

def write_table(f, errors, sub, val):
    f.write(r"\begin{tabular}{|c||ccccc|} " + "\n")
    f.write(r"\multicolumn{6}{c}{Relative test error $E_{T," + sub + r"}^{\%}$ in \% against " + val + r"} \\" + "\n")
    f.write(r"\hline " + "\n")
    f.write(r"\diagbox{Epochs $N_e$}{Samples $N_s$} & " +
            " & ".join([str(int(n_s)) for n_s in arr_n_s]) +
            r" \\ " +  "\n")
    f.write(r"\hline\hline " + "\n")
    for i_tf_e, tf_e in enumerate(arr_tf_epochs):
        f.write(f"{int(tf_e)} " + r"& $" + 
                "$ & $".join(
                    [f"{e:.2f}" for e in errors[:, i_tf_e]]
                ) +
                r"$ \\ " + "\n")
    f.write(r"\hline" + "\n") 
    f.write(r"\end{tabular}" + "\n")

with open(os.path.join(EQN_PATH, "results", "systematic", "tables.tex"),
          "w+") as f:
    write_table(f, errors_test_mean, "m", "mean")
    write_table(f, errors_test_std, r"\sigma", "standard deviation")
