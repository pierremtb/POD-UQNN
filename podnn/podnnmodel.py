"""Module declaring a class for a POD-NN model."""

import os
import sys
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pyDOE import lhs

from .pod import get_pod_bases
from .handling import pack_layers
from .logger import Logger
from .neuralnetwork import NeuralNetwork

SETUP_DATA_NAME = "setup_data.pkl"
TRAIN_DATA_NAME = "train_data.pkl"
MODEL_NAME = "model.h5"

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

        self.save_dir = save_dir
        self.setup_data_path = os.path.join(save_dir, SETUP_DATA_NAME)
        self.train_data_path = os.path.join(save_dir, TRAIN_DATA_NAME)
        self.model_path = os.path.join(save_dir, MODEL_NAME)
        # self.model_cache_params_path = os.path.join(save_dir, "model_params.pkl")

        self.regnn = None
        self.V = None

    def u(self, X, t, mu):
        return X[0]*t + mu

    def u_array(self, X, t, mu):
        U = np.zeros((self.n_v, X.shape[1], self.n_t))
        # Initial condition
        for i in range(0, self.n_t):
            U[:, :, i] = self.u(X, t[i], mu)

        return U.reshape((self.n_h, self.n_t))

    def sample_mu(self, n_s, mu_min, mu_max):
        pbar = tqdm(total=100)
        X_lhs = lhs(n_s, len(mu_min)).T
        pbar.update(50)
        mu_lhs = mu_min + (mu_max - mu_min)*X_lhs
        pbar.update(50)
        pbar.close()
        return mu_lhs

    def create_snapshots(self, n_s, n_st, n_d, n_h, mu_lhs,
                         t_min=0, t_max=0):
        n_xyz = self.x_mesh.shape[0]

        # Declaring the output arrays
        X_v = np.zeros((n_st, n_d))
        U = np.zeros((n_h, n_st))

        if self.has_t:
            U_struct = np.zeros((n_h, self.n_t, n_s))

            # Creating the time steps
            t = np.linspace(t_min, t_max, self.n_t)
            tT = t.reshape((self.n_t, 1))

        # Getting the nodes coordinates
        X = self.x_mesh[:, 1:].T

        if self.has_t:
            for i in tqdm(range(n_s)):
                # Getting the snapshot times indices
                s = self.n_t * i
                e = self.n_t * (i 
                + 1)

                # Setting the regression inputs (t, mu)
                X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))

                # Calling the analytical solution function
                U[:, s:e] = self.u_array(X, t, mu_lhs[i, :])
                U_struct[:, :, i] = \
                    np.reshape(U[:, s:e],
                               (self.n_v, n_xyz, self.n_t))
        else:
            for i in tqdm(range(n_s)):
                X_v[i, :] = mu_lhs[i]
                U[:, i] = self.u(X, 0, mu_lhs[i, :])
            U_struct = U
        return X_v, U, U_struct

    def split_dataset(self, X_v, v, train_val_ratio, n_st):
        n_st_train = int(train_val_ratio * n_st)
        X_v_train, v_train = X_v[:n_st_train, :], v[:n_st_train, :]
        X_v_val, v_val = X_v[n_st_train:, :], v[n_st_train:, :]
        return X_v_train, v_train, X_v_val, v_val

    def convert_dataset(self, u_mesh, X_v, train_val_ratio, eps, eps_init=None,
                        use_cache=False, save_cache=False):
        if use_cache and os.path.exists(self.train_data_path):
            data = self.load_train_data()
            self.V = data[0]
            return data[1:]

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
            self.V = get_pod_bases(U.reshape((n_h, self.n_t, n_s)),
                                   eps, eps_init_step=eps_init)
        else:
            self.V = get_pod_bases(U, eps)

        # Projecting
        v = (self.V.T.dot(U)).T

        # Splitting the dataset (X_v, v)
        X_v_train, v_train, X_v_val, v_val = self.split_dataset(
            X_v, v, train_val_ratio, n_s)

        # Creating the validation snapshots matrix
        U_val = self.V.dot(v_val.T)

        if save_cache:
            self.save_train_data(X_v_train, v_train, X_v_val, v_val, U_val)

        return X_v_train, v_train, X_v_val, v_val, U_val

    def generate_dataset(self, mu_min, mu_max, n_s,
                         train_val_ratio, eps, eps_init=None,
                         t_min=0, t_max=0,
                         use_cache=False, save_cache=False):
        if use_cache:
            return self.load_train_data()[1:]

        if self.has_t:
            t_min, t_max = np.array(t_min), np.array(t_max)
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
        print("Doing the LHS sampling on the non-spatial params...")
        mu_lhs = self.sample_mu(n_s, mu_min, mu_max)

        # Creating the snapshots
        print(f"Generating {n_st} corresponding snapshots")
        X_v, U, U_struct = \
            self.create_snapshots(n_s, n_st, n_d, n_h, mu_lhs,
                                  t_min, t_max)

        # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
        # u_rb are the reduced coefficients we're looking for
        if eps_init is not None:
            self.V = get_pod_bases(U_struct, eps, eps_init_step=eps_init)
        else:
            self.V = get_pod_bases(U, eps)

        # Projecting
        v = (self.V.T.dot(U)).T

        # Splitting the dataset (X_v, v)
        X_v_train, v_train, X_v_val, v_val = self.split_dataset(
            X_v, v, train_val_ratio, n_st)

        # Creating the validation snapshots matrix
        U_val = self.V.dot(v_val.T)

        if save_cache:
            self.save_train_data(X_v_train, v_train, X_v_val, v_val, U_val)

        return X_v_train, v_train, X_v_val, v_val, U_val

    def train(self, X_v, v, error_val, layers, epochs, lr, reg_lam,
              frequency=100):
        # Sizes
        n_L = self.V.shape[1]
        n_d = X_v.shape[1]

        # Creating the neural net model, and logger
        # In: (t, mu)
        # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
        layers = pack_layers(n_d, layers, n_L)
        logger = Logger(epochs, frequency)

        self.regnn = tf.keras.Sequential()
        self.regnn.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        for width in layers[1:-1]:
            self.regnn.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal",
                kernel_regularizer=tf.keras.regularizers.l2(reg_lam)))
        self.regnn.add(tf.keras.layers.Dense(
                layers[-1], activation=None,
                kernel_initializer="glorot_normal",
                kernel_regularizer=tf.keras.regularizers.l2(reg_lam)))

        optimizer = tf.keras.optimizers.Adam(lr=1e-1, decay=1e-2)

        self.regnn.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['mse'])

        self.regnn.summary()

        # Setting the error function
        logger.set_error_fn(error_val)

        class LoggerCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                logger.log_train_epoch(epoch, logs['loss'])

        history = self.regnn.fit(
            X_v, v,
            epochs=epochs, validation_split=0., verbose=0,
            callbacks=[LoggerCallback()])

        # Training
        # ub = np.amax(X_v, axis=0)
        # lb = np.amin(X_v, axis=0)
        # self.regnn.fit(X_v, v, lr, epochs, logger, lb=None, ub=None)

        # Saving
        self.save_model()

        return self.regnn

    def restruct(self, U):
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
        tup = (self.n_xyz,)
        if self.has_t:
            tup += (self.n_t,)
        return (self.n_v,) + tup

    def predict_v(self, X_v):
        """Returns the predicted POD projection coefficients."""
        v_pred = self.regnn.predict(X_v)
        return v_pred

    def predict(self, X_v):
        """Returns the predicted solutions, via proj coefficients."""
        # Predicting the projection coefficients
        v_pred = self.predict_v(X_v) 

        # Retrieving the function with the predicted coefficients
        U_pred = self.V.dot(v_pred.T)

        return U_pred

    def save_setup_data(self, X_v_train, v_train, X_v_val, v_val, U_val):
        with open(self.setup_data_path, "wb") as f:
            pickle.dump((self.n_v, self.x_mesh, self.n_t), f)

    def load_train_data(self):
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError("Can't find train data.")
        with open(self.train_data_path, "rb") as f:
            print("Loading train data")
            return pickle.load(f)

    def save_train_data(self, X_v_train, v_train, X_v_val, v_val, U_val):
        with open(self.train_data_path, "wb") as f:
            pickle.dump((self.V, X_v_train, v_train,
                         X_v_val, v_val, U_val), f)

    def load_model(self):
        """Load a (trained) model and params."""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Can't find cached model.")
        # if not os.path.exists(self.model_cache_params_path):
        #     raise FileNotFoundError("Can't find cached model params.")

        print(f"Loading model from {self.model_path}...")
        # print(f"Loading model params from {self.model_cache_params_path}...")
        # with open(self.model_cache_params_path, "rb") as f:
            # layers, reg_lam = pickle.load(f)
        self.regnn = tf.keras.models.load_model(self.model_path)

    def save_model(self):
        tf.keras.models.save_model(self.regnn, self.model_path)
        # self.regnn.save_to(self.model_cache_path, self.model_cache_params_path)

    def save(self):
        self.save_setup_data()
        self.save_train_data()
        self.save_model()

    @classmethod
    def load(cls, save_dir):
        n_v, x_mesh, n_t = PodnnModel.load_setup_data(save_dir)
        podnnmodel = cls(save_dir, n_v, x_mesh, n_t)
        V, _, _, _, _, _ = podnnmodel.load_train_data()
        podnnmodel.V = V
        return podnnmodel

    @classmethod
    def load_setup_data(cls, save_dir):
        setup_data_path = os.path.join(save_dir, SETUP_DATA_NAME)
        if not os.path.exists(setup_data_path):
            raise FileNotFoundError("Can't find setup data.")
        with open(setup_data_path, "rb") as f:
            print("Loading setup data")
            return pickle.load(f)