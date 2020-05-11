
"""Module with a class defining an Artificial Neural Network."""

import os
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import neupy.layers as npl
import neupy.algorithms as npa

NORM_NONE = "none"
NORM_MEANSTD = "meanstd"
NORM_CENTER = "center"

class NeuralNetwork:
    def __init__(self, layers, lr, lam, norm=NORM_NONE, model=None, norm_bounds=None):
        # Making sure the dtype is consistent
        self.dtype = "float64"

        # Setting up optimizer
        self.tf_optimizer = tf.keras.optimizers.Adam(lr)

        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)
        if model is None:
            network = npl.Input(layers[0])
            for width in layers[1:-1]:
                # network = network >> npl.Relu(width)
                network = network >> npl.Tanh(width)
            network = network >> npl.Linear(layers[-1])
            network.show()
            self.model = npa.LevenbergMarquardt(network, step=npa.l2(decay_rate=0.01), mu=0.2, mu_update_factor=1.1, show_epoch=1, verbose=True,
                                                )
            # self.model = npa.LevenbergMarquardt(network, show_epoch=1, verbose=True)
            # self.model = npa.Adam(network, show_epoch=10, verbose=True)

            # self.model = tf.keras.Sequential()
            # self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
            # for width in layers[1:-1]:
            #     self.model.add(tf.keras.layers.Dense(width, tf.nn.tanh))
            # self.model.add(tf.keras.layers.Dense(layers[-1], None))
            # self.model.compile(optimizer=self.tf_optimizer, loss="mse")
            # self.model.summary()
        else:
            self.model = model

        self.batch_size = 0

        self.layers = layers
        self.adv_eps = 1e-2
        self.lr = lr
        self.lam = lam
        self.norm_bounds = norm_bounds
        self.norm = norm

        self.logger = None

    def set_normalize_bounds(self, X):
        """Setting the normalization bounds, according to the chosen method."""
        if self.norm == NORM_CENTER:
            lb = np.amin(X, axis=0)
            ub = np.amax(X, axis=0)
            self.norm_bounds = (lb, ub)
        elif self.norm == NORM_MEANSTD:
            lb = X.mean(0)
            ub = X.std(0)
            self.norm_bounds = (lb, ub)

    def normalize(self, X):
        """Perform the normalization on the inputs."""
        if self.norm_bounds is None:
            return X
        if self.norm == NORM_CENTER:
            lb, ub = self.norm_bounds
            X = (X - lb) - 0.5 * (ub - lb)
        elif self.norm == NORM_MEANSTD:
            mean, std = self.norm_bounds
            X = (X - mean) / std
        return X

    def fit(self, X_v, v, X_v_val=None, v_val=None, epochs=1000):
        """Train the model over a given dataset, and parameters."""

        # Normalizing and preparing inputs
        self.set_normalize_bounds(X_v)
        X_v = self.normalize(X_v)
        X_v_val = self.normalize(X_v_val)
        print("Normalized done")

        self.model.train(X_v, v, X_test=X_v_val, y_test=v_val, epochs=epochs)
        # self.model.train(X_v, v, epochs=epochs)
        self.model.plot_errors()

    def predict(self, X):
        """Get the prediction for a new input X."""
        X = self.normalize(X)
        return self.model.predict(X)

    def summary(self):
        """Print a summary of the TensorFlow/Keras model."""
        return

    def save_to(self, model_path, params_path):
        return
        """Save the (trained) model and params for later use."""
        # with open(params_path, "wb") as f:
        #     pickle.dump((self.layers, self.lr, self.lam, self.norm, self.norm_bounds), f)
        # tf.keras.models.save_model(self.model, model_path)

    @classmethod
    def load_from(cls, model_path, params_path):
        """Load a (trained) model and params."""

        # if not os.path.exists(model_path):
        #     raise FileNotFoundError("Can't find cached model.")
        # if not os.path.exists(params_path):
        #     raise FileNotFoundError("Can't find cached model params.")

        # print(f"Loading model from {model_path}")
        # with open(params_path, "rb") as f:
        #     layers, lr, lam, norm, norm_bounds = pickle.load(f)
        # print(f"Loading model params from {params_path}")
        # model = tf.keras.models.load_model(model_path)
        # return cls(layers, lr, lam, model=model, norm=norm, norm_bounds=norm_bounds)

