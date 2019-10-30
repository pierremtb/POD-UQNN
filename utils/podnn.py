import numpy as np
import tensorflow as tf
import os
import sys
from tqdm import tqdm
import pickle
from pyDOE import lhs

sys.path.append("utils")
from pod import get_pod_bases
from handling import pack_layers
from logger import Logger
from neuralnetwork import NeuralNetwork


class PodnnModel(object):
    def __init__(self, n_v, n_x, n_t, eqnPath):
        # Dimension of the function output
        self.n_v = n_v
        # List of number of nodes in each direction of x
        self.n_x = n_x
        # Number of time steps
        self.n_t = n_t

        self.eqnPath = eqnPath

    def auto_meshgrid(self, x, t):
        X0, *XT = np.meshgrid(*x, t)
        X = [X0]
        if len(XT) > 1: 
            X.append(XT[:-1])
        T = XT[-1]
        return X, T

    def u(self, X, t, mu):
        return

    def u_array(self, x, t, mu):
        X, T = self.auto_meshgrid(x, t)
        U = np.zeros(self.get_n_x_tuple() + (self.n_t,))
        U = self.u(X, T, mu)
        return U

    def sample_mu(self, n_s, mu_min, mu_max):
        pbar = tqdm(total=100)
        X_lhs = lhs(n_s, len(mu_min)).T
        pbar.update(50)
        mu_lhs = mu_min + (mu_max - mu_min)*X_lhs
        pbar.update(50)
        pbar.close()
        return mu_lhs

    def get_n_x_tuple(self):
        tup = ()
        for n_x_i in self.n_x:
            tup += (n_x_i,)
        return tup

    def create_snapshots(self, n_s, nn_s, n_d, n_h,
                         x_min, x_max, t_min, t_max, mu_lhs):
        X_v = np.zeros((nn_s, n_d))
        U = np.zeros((n_h, nn_s))
        U_struct = np.zeros(self.get_n_x_tuple() + (self.n_t, n_s))
        x = [np.linspace(x_min[i], x_max[i], self.n_x[i])
             for i in range(len(self.n_x))]
        t = np.linspace(t_min, t_max, self.n_t)
        tT = t.reshape((self.n_t, 1))
        for i in tqdm(range(n_s)):
            # Calling the analytical solution function
            s = self.n_t * i
            e = self.n_t * (i + 1)
            X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))
            U[:, s:e] = np.reshape(self.u_array(x, t, mu_lhs[i, :]),
                                   (n_h, self.n_t))
            U_struct[:, :, i] = np.reshape(U[:, s:e],
                                           self.get_n_x_tuple() + (self.n_t,))
        return X_v, U, U_struct

    def split_dataset(self, X_v, v, train_val_ratio, nn_s):
        nn_s_train = int(train_val_ratio * nn_s)
        X_v_train, v_train = X_v[:nn_s_train, :], v[:nn_s_train, :]
        X_v_val, v_val = X_v[nn_s_train:, :], v[nn_s_train:, :]
        return X_v_train, v_train, X_v_val, v_val

    def generate_dataset(self, x_min, x_max, t_min, t_max,
                         mu_min, mu_max, n_s,
                         train_val_ratio, eps, eps_init=None,
                         use_cache=False, save_cache=False):
        
        cache_path = os.path.join(self.eqnPath, "cache", "prep_data.pkl")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                print("Loaded cached data")
                return pickle.load(f)

        x_min, x_max = np.array(x_min), np.array(x_max)
        t_min, t_max = np.array(t_min), np.array(x_max)
        mu_min, mu_max = np.array(mu_min), np.array(mu_max)

        # Total number of snapshots
        nn_s = self.n_t * n_s 
        # Number of input in time (1) + number of params
        n_d = 1 + len(mu_min)
        # Number of DOFs
        n_h = self.n_v * np.prod(self.n_x)

        # LHS sampling (first uniform, then perturbated)
        print("Doing the LHS sampling on the non-spatial params...")
        mu_lhs = self.sample_mu(n_s, mu_min, mu_max)

        # Creating the snapshots
        print(f"Generating {nn_s} corresponding snapshots")
        X_v, U, U_struct = self.create_snapshots(n_s, nn_s, n_d, n_h,
                                                 x_min, x_max, t_min, t_max, mu_lhs)

        # Getting the POD bases, with u_L(x, mu) = V.u_rb(x, mu) ~= u_h(x, mu)
        # u_rb are the reduced coefficients we're looking for
        if eps_init is not None:
            self.V = get_pod_bases(U_struct, eps, eps_init_step=eps_init)
        else:
            self.V = get_pod_bases(U, eps)

        # Projecting
        v = (self.V.T.dot(U)).T
       
        # Splitting the dataset (X_v, v)
        X_v_train, v_train, X_v_val, v_val = self.split_dataset(X_v, v, train_val_ratio, nn_s)
       
        # Creating the validation snapshots matrix
        U_val = self.V.dot(v_val.T)

        # if save_cache:
        #     with open(cache_path, "wb") as f:
        #         pickle.dump((X_v_train, v_train, X_v_val, v_val,
        #                      lb, ub, V, U_val), f)

        return X_v_train, v_train, X_v_val, v_val, U_val

    def train(self, X_v, v, error_val, layers, epochs, lr, lam, frequency=1000):
        # Sizes
        n_L = self.V.shape[1]
        n_d = X_v.shape[1]

        # Creating the neural net model, and logger
        # In: (t, mu)
        # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
        layers = pack_layers(n_d, layers, n_L)
        logger = Logger(epochs, 100)
        self.regnn = NeuralNetwork(layers, lr, epochs, lam, logger)

        # Setting the error function
        logger.set_error_fn(error_val)

        # Training
        self.regnn.fit(X_v, v, epochs)

        # Saving
        self.regnn.save_to(os.path.join(self.eqnPath, "cache", "model.h5"))
        
        return self.regnn

    def restruct(self, U):
        print(U.shape)
        n_s = int(U.shape[1] / self.n_t)
        U_struct = np.reshape(U, self.get_n_x_tuple() + (self.n_t, n_s)) 
        return U_struct

    def predict(self, X_v_val):
        v_pred = self.regnn.predict(X_v_val)

        # Retrieving the function with the predicted coefficients
        U_pred = self.V.dot(v_pred.T)

        return U_pred

