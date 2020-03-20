#%%
import os
import sys
sys.path.append(os.path.join("..", ".."))
from poduqnn.plotting import figsize, savefig

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
#%%
N_tst = 300
x_tst = np.linspace(-6, 6, N_tst).reshape(-1, 1)
D = 1
y_tst = x_tst**3

N = 20
lb = int(2/(2*6) * N_tst)
ub = int((2+2*4)/(2*6) * N_tst)
# idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        91, 100, 188,   8,  63,  98,  78])
x = x_tst[lb + idx]
y = y_tst[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 9
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])

#%%
from poduqnn.varneuralnetwork import VarNeuralNetwork

layers = [1, 20, 20, 1]
M = 5
u_pred_samples = np.zeros((M, y_tst.shape[0], y_tst.shape[1]))
u_pred_var_samples = np.zeros_like(u_pred_samples)

for i in range(5):
    model = VarNeuralNetwork(layers, lr=0.01, lam=0.001, norm="minmax")
    model.fit_simple(x, y, epochs=5000)
    u_pred_samples[i], u_pred_var_samples[i] = model.predict(x_tst)

u_pred = u_pred_samples.mean(0)
u_pred_var = (u_pred_var_samples + u_pred_samples ** 2).mean(0) - u_pred ** 2
u_pred_sig = np.sqrt(u_pred_var)

#%% Predictions and plotting
lower = u_pred - 2 * u_pred_sig
upper = u_pred + 2 * u_pred_sig

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_tst.ravel(), upper.ravel(), lower.ravel(), 
                    facecolor='C0', alpha=0.3, label=r"$2\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.plot(x_tst, u_pred, label=r"$\hat{u}_*(x)$")
plt.scatter(x, y, c="r", label=r"$u_T(x)$")
plt.plot(x_tst, y_tst, "r--", label=r"$u_*(x)$")
plt.ylim((y_tst.min(), y_tst.max()))
plt.xlabel("$x$")
plt.legend()
plt.tight_layout()
plt.savefig(f"uq-toy-ensnn.pdf", bbox_inches='tight', pad_inches=0)

# %%
