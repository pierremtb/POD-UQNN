"""Module declaring a class for a POD-NN model."""

import os
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import numba as nb

from .pod import perform_pod, perform_fast_pod
from .logger import Logger
from .custombnn import BayesianNeuralNetwork, NORM_MEANSTD
from .acceleration import loop_u, loop_u_t
from .metrics import re_s
from .handling import split_dataset, sample_mu

SETUP_DATA_NAME = "setup_data.pkl"
TRAIN_DATA_NAME = "train_data.pkl"
MODEL_NAME = "model.h5"
MODEL_PARAMS_NAME = "model_params.pkl"


class PodnnModel:
    """Wrapper class to handle POD projections and regression model."""
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
        self.model_path = os.path.join(resdir, MODEL_NAME)

        self.regnn = None
        self.n_L = None
        self.n_d = None
        self.V = None
        self.layers = None
        self.pod_sig = None

        self.dtype = "float64"
        tf.keras.backend.set_floatx(self.dtype)
        self.save_setup_data()

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

    def split_dataset(self, X_v, v, test_size):
        """Randomly splitting the dataset (X_v, v)."""
        indices = np.random.permutation(X_v.shape[0])
        limit = np.floor(X_v.shape[0] * (1 - test_size)).astype(int)
        train_idx, tst_idx = indices[:limit], indices[limit:]
        return X_v[train_idx], X_v[tst_idx], v[train_idx], v[tst_idx]

    def create_snapshots(self, n_d, n_h, u, mu_lhs,
                         t_min=0, t_max=0, u_noise=0., x_noise=0.):
        """Create a generated snapshots matrix and inputs for benchmarks."""
        n_s = mu_lhs.shape[0]
        n_xyz = self.x_mesh.shape[0]
        n_st = n_s
        if self.has_t:
            n_st *= self.n_t

        # Numba-ifying the function
        u = nb.njit(u)

        # Getting the nodes coordinates
        X = self.x_mesh[:, 1:].T

        # Declaring the common output arrays
        X_v = np.zeros((n_st, n_d))
        U = np.zeros((n_h, n_st))

        # Computing
        U_no_noise = np.zeros((n_h, n_st))
        if self.has_t:
            U_struct = np.zeros((n_h, self.n_t, n_s))
            return loop_u_t(u, self.n_t, self.n_v, n_xyz, n_h,
                            X_v, U, U_no_noise, U_struct, X, mu_lhs, t_min, t_max,
                            u_noise, x_noise)

        return loop_u(u, n_h, X_v, U, U_no_noise, X, mu_lhs, u_noise, x_noise)

    def convert_multigpu_data(self, U_struct, X_v, train_val, eps,
                              eps_init=None, n_L=0):
        """Convert spatial mesh/solution to usable inputs/snapshot matrix.
           U is (n_v, n_xyz, n_t, n_s)."""
        self.n_d = X_v.shape[1]
        self.n_xyz = self.x_mesh.shape[0]
        self.n_h = self.n_xyz * self.n_v
        n_s = U_struct.shape[-1]
        n_t = self.n_t
        if n_t == 0:
            n_t = 1
        
        # Getting random indices to manually do the split
        idx_s = np.random.permutation(n_s)
        limit = np.floor(n_s * (1. - train_val[1])).astype(int)
        train_idx, val_idx = idx_s[:limit], idx_s[limit:]

        # Splitting the struct matrix
        U_train_s = U_struct[..., train_idx]
        U_val_s = U_struct[..., val_idx]

        # Splitting the n_st-sized inputs
        X_v_train = np.zeros((len(train_idx)*n_t, X_v.shape[1]))
        X_v_val = np.zeros((len(val_idx)*n_t, X_v.shape[1]))
        for i, idx in enumerate(train_idx):
            X_v_train[i*n_t:(i+1)*n_t] = X_v[idx*n_t:(idx+1)*n_t]
        for i, idx in enumerate(val_idx):
            X_v_val[i*n_t:(i+1)*n_t] = X_v[idx*n_t:(idx+1)*n_t]

        # Reshaping manually
        U_train = self.destruct(U_train_s) 
        U_val = self.destruct(U_val_s) 

        # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
        # u_rb are the reduced coefficients we're looking for
        if eps_init is not None and self.has_t:
            # Never tested
            n_s = int(n_s / self.n_t)
            self.V = perform_fast_pod(U_train.reshape((self.n_h, self.n_t, n_s)),
                                      eps, eps_init)
        else:
            self.V = perform_pod(U_train, eps, n_L, True)
        self.n_L = self.V.shape[1]

        # Projecting
        v_train = self.project_to_v(U_train)
        v_val = self.project_to_v(U_val)
        
        # Checking the POD error
        U_pod = self.V.dot(v_train.T)
        self.pod_sig = np.stack((U_train, U_pod), axis=-1).std(-1).mean(-1)
        print(f"Mean pod sig: {self.pod_sig.mean()}")

        self.save_train_data(X_v_train, v_train, U_train, X_v_val, v_val, U_val)
        return X_v_train, v_train, X_v_val, v_val, U_val

    def generate_dataset(self, u, mu_min, mu_max, n_s,
                         train_val, eps=0., eps_init=None, n_L=0,
                         t_min=0, t_max=0, u_noise=0., x_noise=0.):
        """Generate a training dataset for benchmark problems."""
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
        mu_lhs = sample_mu(n_s, mu_min, mu_max)
        fake_x = np.zeros_like(mu_lhs)

        _, _, mu_lhs_train, mu_lhs_test = \
             self.split_dataset(fake_x, mu_lhs, train_val[1])

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
            self.V = perform_pod(U_train, eps, n_L, False)
        else:
            self.V = perform_fast_pod(U_train_struct, eps, eps_init)
        self.n_L = self.V.shape[1]

        # Projecting
        v_train = (self.V.T.dot(U_train)).T
        v_test = (self.V.T.dot(U_test)).T

        # Saving the POD error
        U_train_pod = self.V.dot(v_train.T)
        self.pod_sig = np.stack((U_train, U_train_pod), axis=-1).std(-1).mean(-1)

        self.save_train_data(X_v_train, v_train, U_train, X_v_test, v_test, U_test)
        return X_v_train, v_train, U_train, X_v_test, v_test, U_test

    def initBNN(self, h_layers, lr, klw, 
                pi_1=1., pi_2=0.1, norm=NORM_MEANSTD):
        """Init the Bayesian Neural Network regression model."""
        self.layers = [self.n_d, *h_layers, self.n_L]
        self.regnn = BayesianNeuralNetwork(self.layers, lr, klw,
                                           pi_1=pi_1, pi_2=pi_2,
                                           norm=norm)
        self.regnn.summary()

    def train(self, X_v, v, X_v_val, v_val, epochs,
              freq=100, silent=False, X_out=None):
        """Train the POD-NN's regression model, and save it."""
        if self.regnn is None:
            raise ValueError("Regression model isn't defined.")

        # Validation and logging
        logger = Logger(epochs, freq, silent=silent)
        def err_fn():
            v_pred, v_pred_sig = self.predict_v(X_v_val)
            log = {
                "RE_v": re_s(v_val.T, v_pred.T),
                "std": tf.reduce_sum(v_pred.std(0)),
                "in": tf.reduce_mean(v_pred_sig)
            }
            if X_out is not None:
                _, sig_out = self.predict_v(X_out)
                log["ino"] = tf.reduce_mean(sig_out),
            return log
        logger.set_val_err_fn(err_fn)

        # Training
        self.regnn.fit(X_v, v, epochs, logger)

        # Saving
        self.save_model()
        return logger.get_logs()

    def predict_dist(self, X, samples=100):
        """Approximate the distribution on U from the one on v."""
        v_dist = self.regnn.predict_dist(X)
        U_sum = np.zeros((self.n_h, X.shape[0]))
        U_sum_sq = np.zeros_like(U_sum)
        for i in range(samples):
            U_i = self.project_to_U(v_dist.sample().numpy())
            U_sum += U_i
            U_sum_sq += U_i**2
        U_pred = U_sum / samples
        U_pred_sig = np.sqrt((samples * U_sum_sq - U_sum**2) \
                     / (samples * (samples - 1)))
        return U_pred, U_pred_sig

    def predict_v(self, X, samples=5):
        """Predict the projection coefficients (regression outputs)."""
        v_pred_samples = np.zeros((samples, X.shape[0], self.n_L))
        v_pred_sig_samples = np.zeros((samples, X.shape[0], self.n_L))

        for i in range(samples):
            # v_pred_samples[i], v_pred_sig_samples[i] = self.predict_dist(X)
            v_dist = self.regnn.predict_dist(X)
            v_pred_samples[i] = v_dist.mean().numpy()
            v_pred_sig_samples[i] = v_dist.variance().numpy()

        # Approximate the mixture in a single Gaussian distribution
        v_pred = v_pred_samples.mean(0)
        v_pred_var = (v_pred_sig_samples**2 + v_pred_samples ** 2).mean(0) - v_pred ** 2
        v_pred_sig = np.sqrt(v_pred_var)
        return v_pred.astype(self.dtype), v_pred_sig.astype(self.dtype)

    def predict(self, X, samples=5):
        """Predict the expanded solution."""
        U_pred_samples = np.zeros((samples, self.n_h, X.shape[0]))
        U_pred_sig_samples = np.zeros((samples, self.n_h, X.shape[0]))

        print(f"Averaging {samples} model configurations...")
        for i in tqdm(range(samples)):
            # v_dist = self.regnn.predict_dist(X)
            # v_pred, v_pred_var = v_dist.mean().numpy(), v_dist.variance().numpy()
            U_pred_samples[i], U_pred_sig_samples[i] = self.predict_dist(X)
            # U_pred_samples[i] = self.project_to_U(v_pred)
            # U_pred_sig_samples[i] = self.project_to_U(np.sqrt(v_pred_var))

        # Approximate the mixture in a single Gaussian distribution
        U_pred = U_pred_samples.mean(0)
        U_pred_var = (U_pred_sig_samples**2 + U_pred_samples ** 2).mean(0) \
                      - U_pred ** 2
        U_pred_sig = np.sqrt(U_pred_var)
        return U_pred, U_pred_sig

    def restruct(self, U, no_s=False):
        """Restruct the snapshots matrix DOFs/space-wise and time/snapshots-wise."""
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
        """Destruct the snapshots matrix DOFs/space-wise and time/snapshots-wise."""
        if self.has_t:
            # (n_v, n_xyz, n_t, n_s) -> (n_h, n_st)
            n_s = int(U_struct.shape[-1])
            U = np.zeros((self.n_h, self.n_t * n_s))
            for i in range(n_s):
                s = self.n_t * i
                e = self.n_t * (i + 1)
                U[:, s:e] = U_struct[:, :, :, i].reshape((self.n_h, self.n_t))
            return U

        # (n_v, n_xyz, n_s) -> (n_h, n_s)
        n_s = U_struct.shape[-1]
        U = np.zeros((self.n_h, n_s))
        for i in range(n_s):
            U[:, i] = U_struct[:, :, i].reshape((self.n_h))
        return U

    def get_u_tuple(self):
        """Construct solution shape."""
        tup = (self.n_xyz,)
        if self.has_t:
            tup += (self.n_t,)
        return (self.n_v,) + tup

    def project_to_U(self, v):
        """Helper to expand from the projection coeffictients to the solution."""
        return self.V.dot(v.T)

    def project_to_v(self, U):
        """Helper to compress the solution to the projection coeffictients."""
        return (self.V.T.dot(U)).T

    def tensor(self, X):
        """Helper to make sure quantities are tensor of dtype."""
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def load_train_data(self):
        """Load training data, such as datasets, in cache/."""
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError("Can't find train data.")
        with open(self.train_data_path, "rb") as f:
            print("Loading train data")
            data = pickle.load(f)
            self.n_L = data[0]
            self.n_d = data[1]
            self.V = data[2]
            self.pod_sig = data[3]
            return data[4:]

    def save_train_data(self, X_v_train, v_train, U_train, X_v_test, v_test, U_test):
        """Save training data, such as datasets, in cache/."""
        with open(self.train_data_path, "wb") as f:
            pickle.dump((self.n_L, self.n_d, self.V, self.pod_sig,
                         X_v_train, v_train, U_train, X_v_test, v_test, U_test), f)

    def load_model(self):
        """Load the (trained) POD-NN's regression nn and params."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Can't find cached model.")
        if not os.path.exists(self.model_params_path):
            raise FileNotFoundError("Can't find cached model params.")
        self.regnn = BayesianNeuralNetwork.load_from(self.model_path, self.model_params_path)

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
