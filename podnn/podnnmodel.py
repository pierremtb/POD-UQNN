"""Module declaring a class for a POD-NN model."""

import os
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numba as nb

from .pod import perform_pod, perform_fast_pod
from .handling import pack_layers
from .logger import Logger
from .neuralnetwork import NeuralNetwork
from .advneuralnetwork import AdvNeuralNetwork
from .acceleration import loop_vdot, loop_vdot_t, loop_u, loop_u_t, lhs
from .metrics import re


SETUP_DATA_NAME = "setup_data.pkl"
TRAIN_DATA_NAME = "train_data.pkl"
MODEL_NAME = "model.h5"
MODEL_PARAMS_NAME = "model_params.pkl"


class PodnnModel:
    def __init__(self, save_dir, n_v, x_mesh, n_t):
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
        self.save_dir = save_dir
        self.setup_data_path = os.path.join(save_dir, SETUP_DATA_NAME)
        self.train_data_path = os.path.join(save_dir, TRAIN_DATA_NAME)
        self.model_path = os.path.join(save_dir, MODEL_NAME)
        self.model_params_path = os.path.join(save_dir, MODEL_PARAMS_NAME)

        self.regnn = None
        self.n_L = None
        self.n_d = None
        self.V = None
        self.ub = None
        self.lb = None
        self.layers = None

        self.save_setup_data()

        self.dtype = "float64"
        tf.keras.backend.set_floatx(self.dtype)

    def u(self, X, t, mu):
        """Return the function output, it needs to be extended."""
        raise NotImplementedError

    def sample_mu(self, n_s, mu_min, mu_max):
        """Return a LHS sampling between mu_min and mu_max of size n_s."""
        X_lhs = lhs(n_s, mu_min.shape[0]).T
        mu_lhs = mu_min + (mu_max - mu_min)*X_lhs
        return mu_lhs

    def generate_hifi_inputs(self, n_s, mu_min, mu_max, t_min=0, t_max=0):
        """Return large inputs to be used in a HiFi prediction task."""
        if self.has_t:
            t_min, t_max = np.array(t_min), np.array(t_max)
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

    def split_dataset(self, X_v, v, test_size):
        if not self.has_t:
            # Randomly splitting the dataset (X_v, v)
            return train_test_split(X_v, v, test_size=test_size)

        n_st_train = int((1. - test_size) * X_v.shape[0])
        X_v_train, v_train = X_v[:n_st_train, :], v[:n_st_train, :]
        X_v_val, v_val = X_v[n_st_train:, :], v[n_st_train:, :]
        return X_v_train, X_v_val, v_train, v_val

    def create_snapshots(self, n_s, n_st, n_d, n_h, u, mu_lhs,
                         t_min=0, t_max=0):
        """Create a generated snapshots matrix and inputs for benchmarks."""
        n_xyz = self.x_mesh.shape[0]

        # Numba-ifying the function
        u = nb.njit(u)

        # Getting the nodes coordinates
        X = self.x_mesh[:, 1:].T

        # Declaring the common output arrays
        X_v = np.zeros((n_st, n_d))
        U = np.zeros((n_h, n_st))

        if self.has_t:
            U_struct = np.zeros((n_h, self.n_t, n_s))
            return loop_u_t(u, n_s, self.n_t, self.n_v, n_xyz, n_h,
                            X_v, U, U_struct, X, mu_lhs, t_min, t_max)

        return loop_u(u, n_s, n_h, X_v, U, X, mu_lhs)

    def convert_dataset(self, u_mesh, X_v, train_val_test, eps, eps_init=None,
                        use_cache=False):
        """Convert spatial mesh/solution to usable inputs/snapshot matrix."""
        if use_cache and os.path.exists(self.train_data_path):
            return self.load_train_data()

        n_xyz = self.x_mesh.shape[0]
        n_h = n_xyz * self.n_v
        n_s = X_v.shape[0]

        # U = u_mesh.reshape(n_h, n_st)
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
            self.V = perform_pod(U, eps, True)

        # Projecting
        v = (self.V.T.dot(U)).T

        # Randomly splitting the dataset (X_v, v)
        X_v_train, X_v_test, v_train, v_test = \
            self.split_dataset(X_v, v, train_val_test[2])

        # Creating the validation snapshots matrix
        U_test = self.V.dot(v_test.T)

        self.save_train_data(X_v, X_v_train, v_train, X_v_test, v_test, U_test)

        return X_v_train, v_train, X_v_test, v_test, U_test

    def generate_dataset(self, u, mu_min, mu_max, n_s,
                         train_val_test, eps, eps_init=None,
                         t_min=0, t_max=0,
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

        # Number of DOFs
        n_h = self.n_v * self.x_mesh.shape[0]

        # LHS sampling (first uniform, then perturbated)
        print("Doing the LHS sampling on the non-spatial params...")
        mu_lhs = self.sample_mu(n_s, mu_min, mu_max)

        # Creating the snapshots
        print(f"Generating {n_st} corresponding snapshots")
        X_v, U, U_struct = \
            self.create_snapshots(n_s, n_st, n_d, n_h, u, mu_lhs,
                                  t_min, t_max)

        # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
        # u_rb are the reduced coefficients we're looking for
        if eps_init is None:
            self.V = perform_pod(U, eps, True)
        else:
            self.V = perform_fast_pod(U_struct, eps, eps_init)

        # Projecting
        v = (self.V.T.dot(U)).T

        # Randomly splitting the dataset (X_v, v)
        X_v_train, X_v_test, v_train, v_test = \
            self.split_dataset(X_v, v, train_val_test[2])

        # Creating the validation snapshots matrix
        U_test = self.V.dot(v_test.T)

        self.save_train_data(X_v, X_v_train, v_train, X_v_test, v_test, U_test)

        return X_v_train, v_train, X_v_test, v_test, U_test

    def tensor(self, X):
        """Convert input into a TensorFlow Tensor with the class dtype."""
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def initNN(self, h_layers, lr, lam):
        """Create the neural net model."""
        self.layers = pack_layers(self.n_d, h_layers, self.n_L)
        # self.regnn = NeuralNetwork(self.layers, lr, lam, lb=self.lb, ub=self.ub)

    def train(self, X_v, v, epochs, train_val_test, freq=100):
        """Train the POD-NN's regression model, and save it."""
        # if self.regnn is None:
        #     raise ValueError("Regression model isn't defined.")

        # Validation and logging
        logger = Logger(epochs, freq)
        val_size = train_val_test[1] / (train_val_test[0] + train_val_test[1])
        X_v_train, X_v_val, v_train, v_val = \
            self.split_dataset(X_v, v, val_size)
        U_val_mean, U_val_std = self.do_vdot(v_val)
        if self.has_t:
            U_val_mean = U_val_mean.mean(-1)
            U_val_std = U_val_std.std(-1)


        hp = {}
        # Dimension of input, output and latent variable
        hp["X_dim"] = self.layers[0]
        hp["Y_dim"] = self.layers[-1]
        hp["T_dim"] = 0
        hp["Z_dim"] = self.layers[0]
        # DeepNNs topologies
        hp["layers_P"] = [hp["X_dim"]+hp["T_dim"] + hp["Z_dim"],
                        50, 50, 50, 50,
                        hp["Y_dim"]]
        hp["layers_Q"] = [hp["X_dim"]+hp["T_dim"] + hp["Y_dim"],
                        50, 50, 50, 50,
                        hp["Z_dim"]]
        hp["layers_T"] = [hp["X_dim"]+hp["T_dim"]+hp["Y_dim"],
                        50, 50, 50,
                        1]
        # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
        hp["tf_epochs"] = epochs
        hp["tf_lr"] = 0.0001
        hp["tf_b1"] = 0.9
        hp["tf_eps"] = None
        # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel)
        # hp["nt_epochs"] = 500
        # hp["nt_lr"] = 1.0
        # hp["nt_ncorr"] = 50
        # Loss coefficients
        hp["lambda"] = 1.5
        hp["beta"] = 1.0
        # MinMax switching
        hp["k1"] = 1
        hp["k2"] = 5
        # Batch size
        # hp["batch_size_u"] = hp["N_i"] + hp["N_b"]
        hp["batch_size_u"] = X_v_train.shape[0]
        hp["batch_size_f"] = X_v_train.shape[0]
        # Logging
        hp["log_frequency"] = freq

        self.regnn = AdvNeuralNetwork(hp, logger, None, None)
        print("SHAPES: ", U_val_mean.shape, U_val_std.shape)
        def get_val_err():
            v_val_pred = self.predict_v(X_v_val)
            U_val_pred_mean, U_val_pred_std = self.do_vdot(v_val_pred)
            if self.has_t:
                U_val_pred_mean = U_val_pred_mean.mean(-1)
                U_val_pred_std = U_val_pred_std.std(-1)
            return {
                # "L_v": self.regnn.loss(v_val, v_val_pred),
                # "REM_v": re(U_val_mean, U_val_pred_mean),
                # "RES_v": re(U_val_std, U_val_pred_std),
                }
        logger.set_val_err_fn(get_val_err)

        # Training
        self.regnn.fit(X_v_train, v_train)
        # self.regnn.fit(X_v_train, v_train, epochs, logger)

        # Saving
        # self.save_model()

        return logger.get_logs()

    def restruct(self, U, no_s=False):
        """Restruct the snapshots matrix DOFs/space-wise and time/snapshots-wise."""
        if no_s:
            return U.reshape(self.get_u_tuple())
        if self.has_t:
            # (n_h, n_st) -> (n_v, n_xyz, n_t, n_s)
            n_s = int(U.shape[-1] / self.n_t)
            U_struct = np.zeros((self.n_v, U.shape[0], self.n_t, n_s))
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

    def get_u_tuple(self):
        """Return solution shape."""
        tup = (self.n_xyz,)
        if self.has_t:
            tup += (self.n_t,)
        return (self.n_v,) + tup

    def predict_v(self, X_v):
        """Return the predicted POD projection coefficients."""
        # v_pred, v_pred_mean = self.regnn.predict(X_v)
        v_pred = self.regnn.predict_sample(X_v).numpy()
        return v_pred.astype(self.dtype)

    def predict(self, X_v):
        """Returns the predicted solutions, via proj coefficients."""
        # v_pred, v_pred_mean = self.predict_v(X_v)
        v_pred = self.predict_v(X_v)

        # Retrieving the function with the predicted coefficients
        U_pred = self.V.dot(v_pred.T)
        return U_pred

    def predict_heavy(self, X_v):
        """Returns the predicted solutions, via proj coefficients (large inputs)."""
        v_pred_hifi = self.predict_v(X_v)
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
        # tup = self.get_u_tuple()
        # return U_pred_hifi_mean.reshape(tup), U_pred_hifi_std.reshape(tup)

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
            self.ub = data[3]
            self.lb = data[4]
            return data[5:]

    def save_train_data(self, X_v, X_v_train, v_train, X_v_test, v_test, U_test):
        """Save training data, such as datasets."""
        # Set dataset dependent params
        self.n_L = self.V.shape[1]
        self.n_d = X_v.shape[1]
        self.lb = np.amin(X_v, axis=0)
        self.ub = np.amax(X_v, axis=0)

        with open(self.train_data_path, "wb") as f:
            pickle.dump((self.n_L, self.n_d, self.V, self.ub, self.lb,
                         X_v_train, v_train, X_v_test, v_test, U_test), f)

    def load_model(self):
        """Load the (trained) POD-NN's regression nn and params."""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Can't find cached model.")
        if not os.path.exists(self.model_params_path):
            raise FileNotFoundError("Can't find cached model params.")

        self.regnn = NeuralNetwork.load_from(self.model_path, self.model_params_path)

    def save_model(self):
        """Save the POD-NN's regression neural network and parameters."""
        self.regnn.save_to(self.model_path, self.model_params_path)

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
        podnnmodel = cls(save_dir, n_v, x_mesh, n_t)
        podnnmodel.load_train_data()
        podnnmodel.load_model()
        return podnnmodel
