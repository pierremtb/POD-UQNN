
"""Module with a class defining an Artificial Neural Network."""

import os
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import normalize as sknormalize
import numpy as np

NORM_NONE = "none"
NORM_MEANSTD = "meanstd"
NORM_CENTER = "center"
NORM_MINMAX = "minmux"

tfk = tf.keras
tfd = tfp.distributions

class MyCallback(tfk.callbacks.Callback):
    def __init__(self, logger):
        self.logger = logger
    def on_epoch_end(self, epoch, logs):
        self.logger.log_train_epoch(epoch, logs["loss"])

class TFPBayesianNeuralNetwork:
    def __init__(self, layers, lr, klw, norm=NORM_NONE, model=None, norm_bounds=None):
        # Making sure the dtype is consistent
        self.dtype = "float64"

        # Setting up optimizer and params
        self.tf_optimizer = tf.keras.optimizers.Adam(lr)
        self.layers = layers
        self.lr = lr
        self.klw = klw
        self.norm_bounds = norm_bounds
        self.logger = None
        self.batch_size = 0
        self.norm = norm

        # Setting up the model
        tf.keras.backend.set_floatx(self.dtype)
        if model is None:
            self.model = self.build_model()
        else:
            self.model = model

    def build_model(self):
        """Descriptive Keras model."""
        def posterior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                                scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                    reinterpreted_batch_ndims=1)),
            ])
            
        def prior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
            ])

        model = tfk.Sequential([
            tfk.layers.InputLayer((self.layers[0],)),
            # *[tfk.layers.Dense(width, activation="softplus")
            #     for width in self.layers[1:-1]],
            *[tfp.layers.DenseVariational(width, posterior, prior,
                                          activation="relu", kl_weight=self.klw)
                for width in self.layers[1:-1]],
            # tfk.layers.Dense(self.layers[-1] * 2),
            tfp.layers.DenseVariational(self.layers[-1] * 2, posterior, prior, kl_weight=self.klw),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :self.layers[-1]],
                                    scale=1e-3 + tf.math.softplus(0.01 * t[..., self.layers[-1]:]))),
        ])

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
            return self.tensor(v)

        # Optimizing
        self.tf_optimization(X_v, v, epochs)
        # self.model.fit(X_v, v, epochs=epochs, verbose=0, callbacks=[MyCallback(logger)])

        self.logger.log_train_end(epochs, np.array(0.))

    def predict(self, X, samples=200):
        """Get the prediction for a new input X."""
        X = self.normalize(X)
        yhats_mean = np.zeros((samples, X.shape[0], self.layers[-1]))
        yhats_var = np.zeros((samples, X.shape[0], self.layers[-1]))
        for i in range(samples):
            dist = self.model(X)
            yhats_mean[i] = dist.mean().numpy()
            yhats_var[i] = dist.variance().numpy()
        yhat = yhats_mean.mean(0)
        yhat_var = (yhats_var + yhats_mean ** 2).mean(0) - yhat ** 2
        return yhat, np.sqrt(yhat_var)

    def summary(self):
        """Print a summary of the TensorFlow/Keras model."""
        return self.model.summary()

    def tensor(self, X):
        """Convert input into a TensorFlow Tensor with the class dtype."""
        return X.astype(self.dtype)
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
