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
    def __init__(self, n_v, x_mesh, n_t, eqnPath):
        # Dimension of the function output
        self.n_v = n_v
        # Mesh definition array in space
        self.x_mesh = x_mesh
        # Number of DOFs
        self.n_h = self.n_v * x_mesh.shape[0]
        # Number of time steps
        self.n_t = n_t
        self.has_t = self.n_t > 0

        self.eqnPath = eqnPath

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

    # def get_x_tuple(self):
    #     tup = (self.n_x,)
    #     if self.has_y:
    #         tup += (self.n_y,)
    #         if self.has_z:
    #             tup += (self.n_z,)
    #     return tup

    # def get_u_tuple(self):
    #     tup = self.get_x_tuple()
    #     return (self.n_v,) + tup

    def create_snapshots(self, n_s, nn_s, n_d, n_h, mu_lhs,
                         t_min=0, t_max=0):
        n_nodes = self.x_mesh.shape[0]
        dim = self.x_mesh.shape[1] - 1

        # Declaring the output arrays
        X_v = np.zeros((nn_s, n_d))
        U = np.zeros((n_h, nn_s))
        
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
                e = self.n_t * (i + 1)

                # Setting the regression inputs (t, mu)
                X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))

                # Calling the analytical solution function
                U[:, s:e] = self.u_array(X, t, mu_lhs[i, :])
                U_struct[:, :, i] = \
                        np.reshape(U[:, s:e],
                                (self.n_v, n_nodes, self.n_t))
        else:
            for i in tqdm(range(n_s)):
                X_v[i, :] = mu_lhs[i]
                U[:, i] = self.u(X, 0, mu_lhs[i, :])
            U_struct = U
        return X_v, U, U_struct

    def split_dataset(self, X_v, v, train_val_ratio, nn_s):
        nn_s_train = int(train_val_ratio * nn_s)
        X_v_train, v_train = X_v[:nn_s_train, :], v[:nn_s_train, :]
        X_v_val, v_val = X_v[nn_s_train:, :], v[nn_s_train:, :]
        return X_v_train, v_train, X_v_val, v_val

    def convert_dataset(self, u_mesh, train_val_ratio, eps, eps_init=None,
                        use_cache=False, save_cache=False):
        


    def generate_dataset(self, mu_min, mu_max, n_s,
                         train_val_ratio, eps, eps_init=None,
                         t_min=0, t_max=0,
                         use_cache=False, save_cache=False):
        
        cache_path = os.path.join(self.eqnPath, "cache", "prep_data.pkl")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                print("Loaded cached data")
                data = pickle.load(f)
                self.V = data[0]
                return data[1:]

        if self.has_t:
            t_min, t_max = np.array(t_min), np.array(t_max)
        mu_min, mu_max = np.array(mu_min), np.array(mu_max)

        # Total number of snapshots
        nn_s = n_s 
        if self.has_t:
            nn_s *= self.n_t

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
        print(f"Generating {nn_s} corresponding snapshots")
        X_v, U, U_struct = \
            self.create_snapshots(n_s, nn_s, n_d, n_h, mu_lhs,
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
        X_v_train, v_train, X_v_val, v_val = self.split_dataset(X_v, v, train_val_ratio, nn_s)
       
        # Creating the validation snapshots matrix
        U_val = self.V.dot(v_val.T)

        if save_cache:
            with open(cache_path, "wb") as f:
                pickle.dump((self.V, X_v_train, v_train, X_v_val, v_val, U_val), f)

        return X_v_train, v_train, X_v_val, v_val, U_val

    def train(self, X_v, v, error_val, layers, epochs, lr, lam,
              frequency=1000):
        # Sizes
        n_L = self.V.shape[1]
        n_d = X_v.shape[1]

        # Creating the neural net model, and logger
        # In: (t, mu)
        # Out: u_rb = (u_rb_1, u_rb_2, ..., u_rb_L)
        layers = pack_layers(n_d, layers, n_L)
        logger = Logger(epochs, frequency)
        self.regnn = NeuralNetwork(layers, lr, epochs, lam, logger)

        # Setting the error function
        logger.set_error_fn(error_val)

        # Training
        self.regnn.fit(X_v, v, epochs)

        # Saving
        self.regnn.save_to(os.path.join(self.eqnPath, "cache", "model.h5"))
        
        return self.regnn

    def restruct(self, U): 
        if self.has_t:
            n_s = int(U.shape[-1] / self.n_t)
            U_struct = np.zeros((U.shape[0], self.n_t, n_s))
            for i in range(n_s):
                s = self.n_t * i
                e = self.n_t * (i + 1)
                U_struct[:, :, i] = U[:, s:e]
            return U_struct
        n_s = U.shape[-1]
        return U.reshape(self.get_u_tuple() + (n_s,))        

    def predict(self, X_v_val):
        v_pred = self.regnn.predict(X_v_val)

        # Retrieving the function with the predicted coefficients
        U_pred = self.V.dot(v_pred.T)

        return U_pred

