
"""Module with a class defining an Artificial Neural Network."""

import os
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .logger import LoggerCallback

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions

NORM_NONE = "none"
NORM_MEANSTD = "meanstd"
NORM_CENTER = "center"
NORM_MINMAX = "minmax"

tfk = tf.keras
tfd = tfp.distributions


class BayesianNeuralNetwork:
    def __init__(self, layers, lr, klw, adv_eps=None, norm=NORM_NONE, model=None, norm_bounds=None):
        # Making sure the dtype is consistent
        self.dtype = "float64"

        # Setting up optimizer and params
        self.tf_optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.layers = layers
        self.lr = lr
        self.klw = klw
        self.norm_bounds = norm_bounds
        self.logger = None
        self.batch_size = 0
        self.norm = norm
        self.adv_eps = adv_eps

        # Setting up the model
        tf.keras.backend.set_floatx(self.dtype)
        if model is None:
            self.model = self.build_model()
        else:
            self.model = model

    def build_model(self):
        """Descriptive Keras model."""

        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tfk.models.Sequential([
                # tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
                tfp.layers.VariableLayer(n, dtype=dtype, trainable=True),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1,
                ))
            ])

        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = tf.math.log(tf.math.expm1(tf.constant(1., dtype=dtype)))
            return tfk.models.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n],
                        scale=1e-5 + 1. * tf.math.softplus(c + t[..., n:]),
                    ),
                    reinterpreted_batch_ndims=1,
                ))
            ])
        n_L = self.layers[-1]
        model = tfk.models.Sequential([
            tfk.layers.InputLayer(self.layers[0]),
            *[
            tfp.layers.DenseVariational(
                units=width,
                activation="relu",
                make_posterior_fn=posterior_mean_field,
                make_prior_fn=prior_trainable,
                kl_weight=self.klw,
                dtype=self.dtype,
            ) for width in self.layers[1:-1]],
            tfp.layers.DenseVariational(
                units=2 * n_L,
                activation="linear",
                make_posterior_fn=posterior_mean_field,
                make_prior_fn=prior_trainable,
                kl_weight=self.klw,
                dtype=self.dtype,
            ),
            tfp.layers.DistributionLambda(lambda t:
                tfd.MultivariateNormalDiag(
                    loc=t[..., :n_L],
                    scale_diag=1e-5 + tf.math.softplus(0.01 * t[..., n_L:]),
                ),
            ),
        ])
        model.compile(loss=lambda y, yhat: -tf.reduce_sum(yhat.log_prob(y)),
                    optimizer=tfk.optimizers.Adam(self.lr))

        return model

    def set_normalize_bounds(self, X):
        if self.norm == NORM_CENTER or self.norm == NORM_MINMAX:
            lb = X.min(0)
            ub = X.max(0)
            self.norm_bounds = (lb, ub)
        elif self.norm == NORM_MEANSTD:
            lb = X.mean(0)
            ub = X.std(0)
            self.norm_bounds = (lb, ub)

    def normalize(self, X):
        if self.norm_bounds is None:
            return self.tensor(X)

        if self.norm == NORM_CENTER:
            lb, ub = self.norm_bounds
            X = (X - lb) - 0.5 * (ub - lb)
        elif self.norm == NORM_MEANSTD:
            mean, std = self.norm_bounds
            X = (X - mean) / std

        return self.tensor(X)

    def fit(self, X_v, v, epochs, logger):
        """Train the model over a given dataset, and parameters."""
        # Setting up logger
        self.logger = logger
        self.logger.log_train_start()

        # Normalizing and preparing inputs
        self.set_normalize_bounds(X_v)
        X_v = self.normalize(X_v)
        v = self.tensor(v)

        # Optimizing
        self.model.fit(X_v, v, epochs=epochs, 
                       verbose=0, callbacks=[LoggerCallback(logger)])

        self.logger.log_train_end(epochs, tf.constant(0., dtype=self.dtype))

    def predict_dist(self, X):
        """Get the prediction distribution for a new input X."""
        X = self.normalize(X)
        return self.model(X)

    def predict(self, X, samples=200):
        """Get the prediction for a new input X, sampled many times."""
        X = self.normalize(X)
        yhat = self.model(X)
        v_pred = np.array([yhat.mean().numpy() for _ in range(samples)]).mean(0)
        return v_pred, np.zeros_like(v_pred)

    def summary(self):
        """Print a summary of the TensorFlow/Keras model."""
        return self.model.summary()

    def tensor(self, X):
        """Convert input into a TensorFlow Tensor with the class dtype."""
        return X.astype("float64")
        # return tf.convert_to_tensor(X, dtype=self.dtype)

    def save_to(self, model_path, params_path):
        """Save the (trained) model and params for later use."""
        with open(params_path, "wb") as f:
            pickle.dump((self.layers, self.lr, self.klw, self.norm, self.norm_bounds), f)
        tf.keras.models.save_model(self.model, model_path)

    @classmethod
    def load_from(cls, model_path, params_path):
        """Load a (trained) model and params."""

        if not os.path.exists(model_path):
            raise FileNotFoundError("Can't find cached model.")
        if not os.path.exists(params_path):
            raise FileNotFoundError("Can't find cached model params.")

        print(f"Loading model from {model_path}")
        with open(params_path, "rb") as f:
            layers, lr, klw, norm, norm_bounds = pickle.load(f)
        print(f"Loading model params from {params_path}")
        model = tf.keras.models.load_model(model_path)
        return cls(layers, lr, klw, model=model, norm=norm, norm_bounds=norm_bounds)
