import matplotlib.pyplot as plt
import sys
import os
import numpy as np

eqnPath = "1d-shekel"
sys.path.append("utils")
from plotting import figsize
sys.path.append(os.path.join("datagen", eqnPath))
from names import X_FILE, U_MEAN_FILE, U_STD_FILE

dirname = os.path.join(eqnPath, "data")
print(f"Reading data to {dirname}")
x = np.load(os.path.join(dirname, X_FILE))
u_mean = np.load(os.path.join(dirname, U_MEAN_FILE))
u_std = np.load(os.path.join(dirname, U_STD_FILE))

fig = plt.figure(figsize=figsize(2, 1))
ax_mean = fig.add_subplot(1, 2, 1)
ax_mean.plot(x, u_mean)
ax_mean.set_title(r"Mean of $u_h(x, \gamma, \beta)$")
ax_mean.set_xlabel("$x$")
ax_std = fig.add_subplot(1, 2, 2)
ax_std.plot(x, u_std)
ax_std.set_title(r"Standard deviation of $u_h(x, \gamma, \beta)$")
ax_std.set_xlabel("$x$")
plt.show()

