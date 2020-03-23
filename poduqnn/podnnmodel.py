"""Module declaring a class for a POD-NN model."""

import os
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import numba as nb
from sklearn.model_selection import train_test_split

from .pod import perform_pod, perform_fast_pod
from .handling import pack_layers
from .logger import Logger
from .neuralnetwork import NeuralNetwork
from .acceleration import loop_u, loop_u_t, lhs
from .metrics import re_s
from .handling import sample_mu, split_dataset


SETUP_DATA_NAME = "setup_data.pkl"
TRAIN_DATA_NAME = "train_data.pkl"
MODEL_NAME = "model.h5"
MODEL_PARAMS_NAME = "model_params.pkl"


class PodnnModel:
    """Wrapper class to handle POD projections and regression model."""
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
        self.layers = None

        self.save_setup_data()

        self.dtype = "float64"
        tf.keras.backend.set_floatx(self.dtype)

    def generate_hifi_inputs(self, n_s, mu_min, mu_max, t_min=0, t_max=0):
        """Return large inputs to be used in a HiFi prediction task."""
        if self.has_t:
            t_min, t_max = np.array(t_min), np.array(t_max)
        mu_min, mu_max = np.array(mu_min), np.array(mu_max)

        mu_lhs = sample_mu(n_s, mu_min, mu_max)

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

    def u_mesh_to_U(self, u_mesh, n_s):
        # Reshaping manually
        U = np.zeros((self.n_h, n_s))
        for i in range(n_s):
            st = self.n_xyz * i
            en = self.n_xyz * (i + 1)
            U[:, i] = u_mesh[st:en, :].T.reshape((self.n_h,))
        return U

    def generate_dataset(self, u, mu_min, mu_max, n_s,
                         train_val, eps, eps_init=None,
                         t_min=0, t_max=0,
                         use_cache=False):
        """Generate a training dataset for benchmark problems."""
        if use_cache:
            return self.load_train_data()

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
        mu_lhs = sample_mu(n_s, mu_min, mu_max)

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
            split_dataset(X_v, v, train_val[-1])

        # Creating the validation snapshots matrix
        U_test = self.V.dot(v_test.T)

        self.save_train_data(X_v, X_v_train, v_train, X_v_test, v_test, U_test)

        return X_v_train, v_train, X_v_test, v_test, U_test

    def initNN(self, h_layers, lr, lam, norm):
        """Create the neural net model."""
        self.layers = pack_layers(self.n_d, h_layers, self.n_L)
        self.regnn = NeuralNetwork(self.layers, lr, lam, norm)

    def train(self, X_v, v, X_v_val, v_val, epochs, freq=100):
        """Train the POD-NN's regression model, and save it."""
        if self.regnn is None:
            raise ValueError("Regression model isn't defined.")

        # Validation and logging
        logger = Logger(epochs, freq)
        def get_val_err():
            v_val_pred = self.predict_v(X_v_val)
            return {
                "L_v": self.regnn.loss(v_val, v_val_pred),
                "RE_v": re_s(v_val.T, v_val_pred.T),
                }
        logger.set_val_err_fn(get_val_err)

        # Training
        self.regnn.fit(X_v, v, epochs, logger)

        # Saving
        self.save_model()
        return logger.get_logs()

    def predict_v(self, X_v):
        """Return the predicted POD projection coefficients."""
        v_pred = self.regnn.predict(X_v).astype(self.dtype)
        return v_pred

    def predict(self, X_v):
        """Returns the predicted solutions, via proj coefficients."""
        v_pred = self.predict_v(X_v)

        # Retrieving the function with the predicted coefficients
        U_pred = self.V.dot(v_pred.T)
        return U_pred

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
        """Construct solution shape."""
        tup = (self.n_xyz,)
        if self.has_t:
            tup += (self.n_t,)
        return (self.n_v,) + tup

    def tensor(self, X):
        """Helper to make sure quantities are tensor of dtype."""
        return tf.convert_to_tensor(X, dtype=self.dtype)

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
            return data[3:]

    def save_train_data(self, X_v, X_v_train, v_train, X_v_test, v_test, U_test):
        """Save training data, such as datasets."""
        # Set dataset dependent params
        self.n_L = self.V.shape[1]
        self.n_d = X_v.shape[1]

        with open(self.train_data_path, "wb") as f:
            pickle.dump((self.n_L, self.n_d, self.V,
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
