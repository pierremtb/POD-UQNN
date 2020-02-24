import numpy as np
import os


arr_n_s = np.loadtxt(os.path.join("results", "systematic", "n_s.csv"))
arr_tf_epochs = np.loadtxt(os.path.join("results", "systematic", "tf_epochs.csv"))
errors_test_mean = np.loadtxt(os.path.join("results", "systematic", "err_t_mean.csv"))
errors_test_std = np.loadtxt(os.path.join("results", "systematic", "err_t_std.csv"))
errors = np.stack((errors_test_mean, errors_test_std), axis=2)


def write_table(f, errors):
    n_err_col = errors.shape[0]
    f.write(r"\begin{tabular}{|c||" + n_err_col*"c" + r"|} " + "\n")
    f.write(r"\multicolumn{" + str(n_err_col + 1) + \
            r"}{c}{Relative hifi test errors $(RE_{\textrm{val}}, RE_{\textrm{tst}})$} \\" + "\n")
    f.write(r"\hline " + "\n")
    f.write(r"\diagbox{Epochs $N_e$}{Samples $N_s$} & " +
            " & ".join([str(int(n_s)) for n_s in arr_n_s]) +
            r" \\ " +  "\n")
    f.write(r"\hline\hline " + "\n")
    for i_tf_e, tf_e in enumerate(arr_tf_epochs):
        f.write(f"{int(tf_e)} " + r"& $" + 
                "$ & $".join(
                    [f"({100*e[0]:.2f}, {100*e[1]:.2f})" for e in errors[:, i_tf_e]]
                ) +
                r"$ \\ " + "\n")
    f.write(r"\hline" + "\n") 
    f.write(r"\end{tabular}" + "\n")


with open(os.path.join("results", "systematic", "tables.tex"),
          "w+") as f:
    write_table(f, errors)
