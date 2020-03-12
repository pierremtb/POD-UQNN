"""Module declaring a class for a POD-NN model."""

import os
import random
import time
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import numba as nb

from .pod import perform_pod, perform_fast_pod
from .handling import pack_layers, split_dataset
from .logger import Logger
from .varneuralnetwork import VarNeuralNetwork, NORM_CENTER, NORM_MEANSTD, NORM_NONE
from .acceleration import loop_vdot, loop_vdot_t, loop_u, loop_u_t, lhs
from .metrics import re, re_s


SETUP_DATA_NAME = "setup_data.pkl"
TRAIN_DATA_NAME = "train_data.pkl"
MODEL_NAME = ("model_p.h5", "model_q.h5", "model_t.h5")
MODEL_PARAMS_NAME = "model_params.pkl"


class PodnnModel:
    def __init__(self, resdir, n_v, x_mesh, n_t):
        # Dimension of the function output
        self.n_v = n_v
        # Mesh definition array in space
        self.x_mesh = x_mesh
        self.n_xyz = x_mesh.shape[0]
        # Number of DOFs
        self.n_h = self.n_v * x_mesh.shape[0]
        # Number of time steps
        self.n_t = n_t
        self.has_t = self.n_t > 0

        # Cache paths
        self.resdir = resdir
        self.setup_data_path = os.path.join(resdir, SETUP_DATA_NAME)
        self.train_data_path = os.path.join(resdir, TRAIN_DATA_NAME)
        self.model_params_path = os.path.join(resdir, MODEL_PARAMS_NAME)
        self.model_path = []

        self.regnn = None
        self.n_L = None
        self.n_d = None
        self.V = None
        self.layers = None

        self.pod_sig = None

        self.save_setup_data()

        self.dtype = "float64"
        tf.keras.backend.set_floatx(self.dtype)

    def u(self, X, t, mu):
        """Return the function output, it needs to be extended."""
        raise NotImplementedError

    def sample_mu(self, n_s, mu_min, mu_max, indices=None):
        """Return a LHS sampling between mu_min and mu_max of size n_s."""
        if indices is not None:
            mu = np.linspace(mu_min, mu_max, n_s)[indices]
            return mu
        X_lhs = lhs(n_s, mu_min.shape[0]).T
        mu_lhs = mu_min + (mu_max - mu_min)*X_lhs
        return mu_lhs

    def generate_hifi_inputs(self, n_s, mu_min, mu_max, t_min=0, t_max=0):
        """Return large inputs to be used in a HiFi prediction task."""
        # if self.has_t:
        #     t_min, t_max = np.array(t_min), np.array(t_max)
        mu_min, mu_max = np.array(mu_min), np.array(mu_max)
        mu_lhs = self.sample_mu(n_s, mu_min, mu_max)

        n_st = n_s
        n_d = mu_min.shape[0]
        if self.has_t:
            n_st *= self.n_t
            n_d += 1

        X_v = np.zeros((n_st, n_d))

        if self.has_t:
            # Creating the time steps
            t = np.linspace(t_min, t_max, self.n_t)
            tT = t.reshape((self.n_t, 1))

            for i in tqdm(range(n_s)):
                # Getting the snapshot times indices
                s = self.n_t * i
                e = self.n_t * (i + 1)

                # Setting the regression inputs (t, mu)
                X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))
        else:
            for i in tqdm(range(n_s)):
                X_v[i, :] = mu_lhs[i]
        return X_v

    def u_mesh_to_U(self, u_mesh, n_s):
        # Reshaping manually
        U = np.zeros((self.n_h, n_s))
        for i in range(n_s):
            st = self.n_xyz * i
            en = self.n_xyz * (i + 1)
            U[:, i] = u_mesh[st:en, :].T.reshape((self.n_h,))
        return U

    def create_snapshots(self, n_d, n_h, u, mu_lhs,
                         t_min=0, t_max=0, u_noise=0., x_noise=0.):
        """Create a generated snapshots matrix and inputs for benchmarks."""
        n_s = mu_lhs.shape[0]
        n_xyz = self.x_mesh.shape[0]

        # Numba-ifying the function
        u = nb.njit(u)

        n_st = n_s
        if self.has_t:
            n_st *= self.n_t

        # Getting the nodes coordinates
        X = self.x_mesh[:, 1:].T

        # Declaring the common output arrays
        X_v = np.zeros((n_st, n_d))
        U = np.zeros((n_h, n_st))

        U_no_noise = np.zeros((n_h, n_st))
        if self.has_t:
            U_struct = np.zeros((n_h, self.n_t, n_s))
            return loop_u_t(u, self.n_t, self.n_v, n_xyz, n_h,
                            X_v, U, U_no_noise, U_struct, X, mu_lhs, t_min, t_max,
                            u_noise, x_noise)

        # U_no_noise = np.zeros((n_h, n_st))
        return loop_u(u, n_h, X_v, U, U_no_noise, X, mu_lhs, u_noise, x_noise)

    def convert_multigpu_data(self, U_struct, X_v, train_val, eps, eps_init=None,
                              n_L=0, use_cache=False, save_cache=False):
        """Convert spatial mesh/solution to usable inputs/snapshot matrix."""
        """U is (n_v, n_xyz, n_t, n_s)."""
        if use_cache and os.path.exists(self.train_data_path):
            return self.load_train_data()

        self.n_xyz = self.x_mesh.shape[0]
        self.n_h = self.n_xyz * self.n_v
        n_st = X_v.shape[0]
        n_s = U_struct.shape[-1]
        n_t = self.n_t

        # Number of input in time (1) + number of params
        self.n_d = X_v.shape[1]
        
        # Getting random indices to manually do the split
        idx_s = np.random.permutation(n_s)
        limit = np.floor(n_s * (1. - train_val[1])).astype(int)
        train_idx, val_idx = idx_s[:limit], idx_s[limit:]
        print(len(train_idx), len(val_idx))

        # Splitting the struct matrix
        U_train_s = U_struct[:, :, :, train_idx]
        U_val_s = U_struct[:, :, :, val_idx]

        # Splitting the n_st-sized inputs
        X_v_train = np.zeros((len(train_idx)*n_t, X_v.shape[1]))
        X_v_val = np.zeros((len(val_idx)*n_t, X_v.shape[1]))
        for i, idx in enumerate(train_idx):
            print(i*n_t,(i+1)*n_t, idx*n_t,(idx+1)*n_t)
            X_v_train[i*n_t:(i+1)*n_t] = X_v[idx*n_t:(idx+1)*n_t]
        for i, idx in enumerate(val_idx):
            X_v_val[i*n_t:(i+1)*n_t] = X_v[idx*n_t:(idx+1)*n_t]
        # X_v_train = X_v[train_idx*self.n_t:train_idx*(self.n_t+1)] 
        # X_v_val = X_v[val_idx*self.n_t:val_idx*(self.n_t+1)] 
        print(X_v.shape, U_struct.shape)
        print(X_v_train.shape, U_train_s.shape)
        print(X_v_val.shape, U_val_s.shape)

        # Reshaping manually
        # U = self.destruct(U_struct) 
        U_train = self.destruct(U_train_s) 
        U_val = self.destruct(U_val_s) 

        # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
        # u_rb are the reduced coefficients we're looking for
        if eps_init is not None and self.has_t:
            # Never tested
            n_s = int(n_s / self.n_t)
            self.V = perform_fast_pod(U.reshape((self.n_h, self.n_t, n_s)),
                                      eps, eps_init)
        else:
            self.V = perform_pod(U_train, eps, n_L, True)

        self.n_L = self.V.shape[1]

        # Projecting
        # v = (self.V.T.dot(U)).T
        v_train = self.project_to_v(U_train)
        v_val = self.project_to_v(U_val)
        
        # Checking the POD error
        U_pod = self.V.dot(v_train.T)
        self.pod_sig = np.stack((U_train, U_pod), axis=-1).std(-1).mean(-1)
        print(f"Mean pod sig: {self.pod_sig.mean()}")

        # Randomly splitting the dataset (X_v, v)
        X_v_train, X_v_val, v_train, v_val = split_dataset(X_v, v, test_size=train_val[1])

        # Creating the validation snapshots matrix
        # U_train = self.V.dot(v_train.T)
        U_val = self.V.dot(v_val.T)

        self.save_train_data(X_v_train, v_train, U_train, X_v_val, v_val, U_val)

        return X_v_train, v_train, X_v_val, v_val, U_val

    def convert_dataset(self, u_mesh, X_v, train_val, eps, eps_init=None,
                        n_L=0, use_cache=False, save_cache=False):
        """Convert spatial mesh/solution to usable inputs/snapshot matrix."""
        if use_cache and os.path.exists(self.train_data_path):
            return self.load_train_data()

        self.n_xyz = self.x_mesh.shape[0]
        n_h = self.n_xyz * self.n_v
        n_s = X_v.shape[0]

        # Number of input in time (1) + number of params
        self.n_d = X_v.shape[1]

        # Reshaping manually
        U = np.zeros((n_h, n_s))
        for i in range(n_s):
            st = self.n_xyz * i
            en = self.n_xyz * (i + 1)
            U[:, i] = u_mesh[st:en, :].T.reshape((n_h,))

        # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
        # u_rb are the reduced coefficients we're looking for
        if eps_init is not None and self.has_t:
            # Never tested
            n_s = int(n_s / self.n_t)
            self.V = perform_fast_pod(U.reshape((n_h, self.n_t, n_s)),
                                      eps, eps_init)
        else:
            self.V = perform_pod(U, eps, n_L, True)

        self.n_L = self.V.shape[1]

        # Projecting
        v = (self.V.T.dot(U)).T
        v = self.project_to_v(U)
        
        # Checking the POD error
        U_pod = self.V.dot(v.T)
        self.pod_sig = np.stack((U, U_pod), axis=-1).std(-1).mean(-1)
        print(f"Mean pod sig: {self.pod_sig.mean()}")

        # Randomly splitting the dataset (X_v, v)
        X_v_train, X_v_val, v_train, v_val = split_dataset(X_v, v, test_size=train_val[1])

        # Creating the validation snapshots matrix
        U_train = self.V.dot(v_train.T)
        U_val = self.V.dot(v_val.T)

        self.save_train_data(X_v_train, v_train, U_train, X_v_val, v_val, U_val)

        return X_v_train, v_train, X_v_val, v_val, U_val

    def generate_dataset(self, u, mu_min, mu_max, n_s,
                         train_val, eps=0., eps_init=None, n_L=0,
                         t_min=0, t_max=0, u_noise=0., x_noise=0.,
                         use_cache=False):
        """Generate a training dataset for benchmark problems."""
        if use_cache:
            return self.load_train_data()

        # if self.has_t:
        #     t_min, t_max = np.array(t_min), np.array(t_max)
        mu_min, mu_max = np.array(mu_min), np.array(mu_max)

        # Total number of snapshots
        n_st = n_s
        if self.has_t:
            n_st *= self.n_t

        # Number of input in time (1) + number of params
        n_d = mu_min.shape[0]
        if self.has_t:
            n_d += 1
        self.n_d = n_d

        # Number of DOFs
        n_h = self.n_v * self.x_mesh.shape[0]

        # LHS sampling (first uniform, then perturbated)
        print("Doing the LHS sampling on the non-spatial params...")
        mu_lhs = self.sample_mu(n_s, mu_min, mu_max)
        fake_x = np.zeros_like(mu_lhs)

        _, _, mu_lhs_train, mu_lhs_test = \
             split_dataset(fake_x, mu_lhs, test_size=train_val[1])

        # Creating the snapshots
        print(f"Generating {n_st} corresponding snapshots")
        X_v_train, U_train, U_train_struct, U_no_noise = \
            self.create_snapshots(n_d, n_h, u, mu_lhs_train,
                                  t_min, t_max, u_noise, x_noise)
        X_v_test, U_test, U_test_struct, _ = \
            self.create_snapshots(n_d, n_h, u, mu_lhs_test,
                                  t_min, t_max)

        # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
        # u_rb are the reduced coefficients we're looking for
        if eps_init is None:
            self.V = perform_pod(U_train, eps=eps, n_L=n_L, verbose=True)
        else:
            self.V = perform_fast_pod(U_train_struct, eps, eps_init)

        self.n_L = self.V.shape[1]

        # Projecting
        v_train = (self.V.T.dot(U_train)).T
        v_test = (self.V.T.dot(U_test)).T

        # Saving the POD error
        U_train_pod = self.V.dot(v_train.T)
        self.pod_sig = np.stack((U_train, U_train_pod), axis=-1).std(-1).mean(-1)
        print(f"Mean pod sig: {self.pod_sig.mean()}")

        # Testing stuff out
        # import matplotlib.pyplot as plt
        # print("n_L: ", self.n_L)
        # x = np.linspace(0, 1.5, 256)
        # U_train_mean = U_train.mean(-1)
        # U_train_pod_mean = U_train_pod.mean(-1)
        # plt.plot(x, U_train_mean, "r--")
        # plt.plot(x, U_train_pod_mean, "b-")
        # lower = U_train_pod_mean - 2 * self.pod_sig
        # upper = U_train_pod_mean + 2 * self.pod_sig
        # plt.fill_between(x, lower, upper, 
        #                  facecolor='orange', alpha=0.5, label=r"$2\textrm{std}(\hat{u}_T(x))$")
        # plt.show()
        # exit(0)
        # import matplotlib.pyplot as plt
        # print("n_L: ", self.n_L)
        # x = np.linspace(-5, 5, 400)
        # u_shape = (1, 400, 400)
        # U_train_pod = self.V.dot(v_train.T)
        # plt.plot(x, U_train[:, 0].reshape(u_shape)[0, :, 199], "b--")
        # plt.plot(x, U_train_pod[:, 0].reshape(u_shape)[0, :, 199], "k,")
        # plt.plot(x, U_train[:, 0].reshape(u_shape)[0, :, 0], "b--")
        # plt.plot(x, U_train_pod[:, 0].reshape(u_shape)[0, :, 0], "k,")
        # # plt.plot(x,U_no_noise[:, 0].reshape(u_shape)[0, :, 199], "r--")
        # plt.show()
        # plt.plot(x, U_train[:, 0].reshape(u_shape)[0, :, 299], "b-")
        # plt.plot(x,U_no_noise[:, 0].reshape(u_shape)[0, :, 299], "r--")
        # plt.show()
        # plt.plot(mu_lhs_train[0], "bx")
        # plt.plot(X_v_train[0], "rx")
        # plt.show()

        self.save_train_data(X_v_train, v_train, U_train, X_v_test, v_test, U_test)

        return X_v_train, v_train, U_train, X_v_test, v_test, U_test

    def tensor(self, X):
        """Convert input into a TensorFlow Tensor with the class dtype."""
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def initNN(self, h_layers, h_layers_t,
               lr, lam, bet, k1, k2, norm=NORM_MEANSTD):
        """Create the neural net model."""
        self.lr = lr
        gan_dims = (self.n_d, self.n_L, self.n_d)
        X_dim, Y_dim, Z_dim = gan_dims
        layers_p = [X_dim+Z_dim, *h_layers, Y_dim]
        layers_q = [X_dim+Y_dim, *h_layers, Z_dim]
        layers_t = [X_dim+Y_dim, *h_layers_t, 1]
        self.layers = (layers_p, layers_q, layers_t)

        self.regnn = [VarNeuralNetwork(self.layers, gan_dims,
                                      lr, lam, bet, k1, k2, norm)]

    def initVNNs(self, n_M, h_layers, lr, lam, adv_eps, norm=NORM_MEANSTD):
        """Create the neural net model."""
        self.lr = lr
        self.layers = [self.n_d, *h_layers, self.n_L]
        self.regnn = []
        self.model_path = []
        for i in range(n_M):
            self.regnn.append(VarNeuralNetwork(self.layers, lr, lam, adv_eps, norm))
            self.model_path.append(os.path.join(self.resdir, f"model-{i}.{time.time()}.h5"))
        self.save_model()

    def train(self, X_v_train, v_train, X_v_val, v_val, epochs, freq=100):
        """Train the POD-NN's regression model, and save it."""
        if self.regnn is None or len(self.regnn) == 0:
            raise ValueError("Regression model isn't defined.")

        U_val = self.project_to_U(v_val)
        U_train = self.project_to_U(v_train)

        logs = []

        for model in self.regnn:
            def get_val_err():
                v_train_pred, _ = model.predict(X_v_train)
                v_val_pred, _ = model.predict(X_v_val)
                U_val_pred = self.project_to_U(v_val_pred)
                U_train_pred = self.project_to_U(v_train_pred)
                return {
                    "MSE": tf.reduce_mean(tf.square(v_train - v_train_pred)),
                    "MSE_V": tf.reduce_mean(tf.square(v_val - v_val_pred)),
                    "RE": re_s(U_train, U_train_pred),
                    "RE_V": re_s(U_val, U_val_pred),
                }
            # Validation, logging, training
            logger = Logger(epochs, freq, silent=True)
            logger.set_val_err_fn(get_val_err)
            model.fit(X_v_train, v_train, epochs, logger)
            logs.append(logger.get_logs())

        # Saving
        self.save_model()

        return logs

    def train_model(self, model_id, X_v_train, v_train, X_v_val, v_val,
                    epochs, freq=100, div_max=False):
        """Train the POD-NN's regression model, and save it."""
        if self.regnn is None or len(self.regnn) == 0:
            raise ValueError("Regression model isn't defined.")

        # U_val = self.project_to_U(v_val)
        # U_train = self.project_to_U(v_train)

        logs = []

        model = self.regnn[model_id]
        def get_val_err():
            v_val_pred, _ = model.predict(X_v_val)
            return {
            #     "MSE_V": tf.reduce_mean(tf.square(v_val - v_val_pred)),
                "RE_V": re_s(v_val.T, v_val_pred.T, div_max),
            }
        # Validation, logging, training
        logger = Logger(epochs, freq)
        logger.set_val_err_fn(get_val_err)
        model.fit(X_v_train, v_train, epochs, logger)
        logs.append(logger.get_logs())

        # # Saving
        # self.save_model()

        # return logs

    def restruct(self, U, no_s=False):
        """Restruct the snapshots matrix DOFs/space-wise and time/snapshots-wise."""
        print(self.get_u_tuple())
        if no_s:
            return U.reshape(self.get_u_tuple())
        if self.has_t:
            # (n_h, n_st) -> (n_v, n_xyz, n_t, n_s)
            n_s = int(U.shape[-1] / self.n_t)
            U_struct = np.zeros((self.n_v, self.n_xyz, self.n_t, n_s))
            for i in range(n_s):
                s = self.n_t * i
                e = self.n_t * (i + 1)
                U_struct[:, :, :, i] = U[:, s:e].reshape(self.get_u_tuple())
            return U_struct

        # (n_h, n_s) -> (n_v, n_xyz, n_s)
        n_s = U.shape[-1]
        U_struct = np.zeros((self.get_u_tuple() + (n_s,)))
        for i in range(n_s):
            U_struct[:, :, i] = U[:, i].reshape(self.get_u_tuple())
        return U_struct

    def destruct(self, U_struct):
        """Restruct the snapshots matrix DOFs/space-wise and time/snapshots-wise."""
        if self.has_t:
            # (n_v, n_xyz, n_t, n_s) -> (n_h, n_st)
            n_s = int(U_struct.shape[-1])
            U = np.zeros((self.n_h, self.n_t * n_s))
            for i in range(n_s):
                s = self.n_t * i
                e = self.n_t * (i + 1)
                U[:, s:e] = U_struct[:, :, :, i].reshape((self.n_h, self.n_t))
            return U

        # (n_h, n_s) -> (n_v, n_xyz, n_s)
        n_s = U.shape[-1]
        U_struct = np.zeros((self.get_u_tuple() + (n_s,)))
        for i in range(n_s):
            U_struct[:, :, i] = U[:, i].reshape(self.get_u_tuple())
        return U_struct

    def get_u_tuple(self):
        """Return solution shape."""
        tup = (self.n_xyz,)
        if self.has_t:
            tup += (self.n_t,)
        return (self.n_v,) + tup

    def predict_v(self, X_v):
        """Return the predicted POD projection coefficients."""
        n_M = len(self.regnn)
        v_pred_samples = np.zeros((X_v.shape[0], self.n_L, n_M))
        v_pred_var_samples = np.zeros((X_v.shape[0], self.n_L, n_M))

        for i, model in enumerate(self.regnn):
            v_pred_samples[:, :, i], v_pred_var_samples[:, :, i] = model.predict(X_v)

        v_pred = v_pred_samples.mean(-1)
        v_pred_var = (v_pred_var_samples + v_pred_samples ** 2).mean(-1) - v_pred ** 2
        v_pred_sig = np.sqrt(v_pred_var)

        return v_pred.astype(self.dtype), v_pred_sig.astype(self.dtype)

    def predict(self, X_v):
        """Return the predicted POD projection coefficients."""
        n_M = len(self.regnn)
        U_pred_samples = np.zeros((self.n_h, X_v.shape[0], n_M))
        U_pred_sig_samples = np.zeros((self.n_h, X_v.shape[0], n_M))

        for i, model in enumerate(self.regnn):
            v_pred, v_pred_var = model.predict(X_v)
            U_pred_samples[:, :, i] = self.project_to_U(v_pred)
            U_pred_sig_samples[:, :, i] = self.project_to_U(np.sqrt(v_pred_var))

        U_pred = U_pred_samples.mean(-1)
        U_pred_var = (U_pred_sig_samples**2 + U_pred_samples ** 2).mean(-1) - U_pred ** 2
        U_pred_sig = np.sqrt(U_pred_var)

        if self.pod_sig is not None:
            U_pred_sig += self.pod_sig[:, np.newaxis]

        return U_pred.astype(self.dtype), U_pred_sig.astype(self.dtype)

    def project_to_U(self, v):
        return self.V.dot(v.T)

    def project_to_v(self, U):
        return (self.V.T.dot(U)).T
    
    def predict_heavy(self, X_v):
        """Returns the predicted solutions, via proj coefficients (large inputs)."""
        v_pred_hifi = self.regnn.predict_sample(X_v)
        return self.do_vdot(v_pred_hifi)

    def do_vdot(self, v):
        """Perform an accelerated dot product, return mean and std."""
        n_s = v.shape[0]
        if self.has_t:
            n_s = int(n_s / self.n_t)
            U_tot = np.zeros((self.n_h, self.n_t))
            U_tot_sq = np.zeros((self.n_h, self.n_t))
            U_tot, U_tot_sq = \
                loop_vdot_t(n_s, self.n_t, U_tot, U_tot_sq, self.V, v)
        else:
            U_tot = np.zeros((self.n_h,))
            U_tot_sq = np.zeros((self.n_h,))
            U_tot, U_tot_sq = loop_vdot(n_s, U_tot, U_tot_sq, self.V, v)

        # Getting the mean and std
        U_pred_hifi_mean = U_tot / n_s
        U_pred_hifi_std = np.sqrt((n_s*U_tot_sq - U_tot**2) / (n_s*(n_s - 1)))
        # Making sure the std has non NaNs
        U_pred_hifi_std = np.nan_to_num(U_pred_hifi_std)

        return U_pred_hifi_mean, U_pred_hifi_std

    def load_train_data(self):
        """Load training data, such as datasets."""
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError("Can't find train data.")
        with open(self.train_data_path, "rb") as f:
            print("Loading train data")
            data = pickle.load(f)
            self.n_L = data[0]
            self.n_d = data[1]
            self.V = data[2]
            self.pod_sig = data[3]
        print(f"Mean pod sig: {self.pod_sig.mean()}")
        return data[4:]

    def save_train_data(self, X_v_train, v_train, U_train, X_v_val, v_val, U_val):
        """Save training data, such as datasets."""

        with open(self.train_data_path, "wb") as f:
            pickle.dump((self.n_L, self.n_d, self.V, self.pod_sig,
                         X_v_train, v_train, U_train, X_v_val, v_val, U_val), f)

    def load_model(self):
        """Load the (trained) POD-NN's regression nn and params."""

        models_exist = True
        for path in self.model_path:
            models_exist = models_exist and os.path.exists(path)
        if not models_exist:
            raise FileNotFoundError("Can't find cached model.")
        if not os.path.exists(self.model_params_path):
            raise FileNotFoundError("Can't find cached model params.")

        self.regnn = []
        for path in self.model_path:
            self.regnn.append(VarNeuralNetwork.load_from(path, self.model_params_path))

    def save_model(self, model_id=-1):
        """Save the POD-NN's regression neural network and parameters."""
        if model_id > -1:
            self.regnn[model_id].save_to(self.model_path[model_id], self.model_params_path)
            return
        # for file in os.listdir(self.resdir):
        #     if file.startswith("model-"):
        #         os.remove(os.path.join(self.resdir, file))
        for i, model in enumerate(self.regnn):
            model.save_to(self.model_path[i], self.model_params_path)

    def save_setup_data(self):
        """Save setup-related data, such as n_v, x_mesh or n_t."""
        with open(self.setup_data_path, "wb") as f:
            pickle.dump((self.n_v, self.x_mesh, self.n_t), f)

    @classmethod
    def load_setup_data(cls, save_dir):
        """Load setup-related data, such as n_v, x_mesh or n_t."""
        setup_data_path = os.path.join(save_dir, SETUP_DATA_NAME)
        if not os.path.exists(setup_data_path):
            raise FileNotFoundError("Can't find setup data.")
        with open(setup_data_path, "rb") as f:
            print("Loading setup data")
            return pickle.load(f)

    @classmethod
    def load(cls, save_dir):
        """Recreate a pre-trained POD-NN model."""
        n_v, x_mesh, n_t = PodnnModel.load_setup_data(save_dir)

        model_path = []
        for file in sorted(os.listdir(save_dir)):
            if file.startswith("model-"):
                model_path.append(os.path.join(save_dir, file))

        podnnmodel = cls(save_dir, n_v, x_mesh, n_t)
        podnnmodel.model_path = model_path
        podnnmodel.load_train_data()
        podnnmodel.load_model()
        return podnnmodel
