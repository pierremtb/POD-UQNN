
"""Module with a class defining an Artificial Neural Network."""

import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import normalize as sknormalize
import numpy as np
from tqdm.auto import tqdm


class NeuralNetwork:
    def __init__(self, layers, lr, lam, model=None, lb=None, ub=None):
        # Making sure the dtype is consistent
        self.dtype = "float64"

        # Setting up optimizer
        self.tf_optimizer = tf.keras.optimizers.Adam(lr)

        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)
        if model is None:
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
            for width in layers[1:-1]:
                self.model.add(tf.keras.layers.Dense(width, tf.nn.tanh))
            self.model.add(tf.keras.layers.Dense(layers[-1], None))
            self.model.compile(optimizer=self.tf_optimizer, loss="mse")
            self.model.summary()
        else:
            self.model = model

        self.batch_size = 0

        self.layers = layers
        self.lr = lr
        self.lam = lam
        self.lb = lb
        self.ub = ub

        self.logger = None

    def normalize(self, X):
        """Apply a kind of normalization to the inputs X."""
        if self.lb is not None and self.ub is not None:
            X = (X - self.lb) - 0.5*(self.ub - self.lb)
            # X = sknormalize(X, norm="max")
            # X = (X - X.mean(0)) / X.std(0)
        return X

    def regularization(self):
        l2_norms = [tf.nn.l2_loss(v) for v in self.wrap_training_variables()]
        l2_norm = tf.reduce_sum(l2_norms)
        return self.lam * l2_norm

    @tf.function
    def loss(self, v, v_pred):
        """Return a MSE loss function between the pred and val."""
        return tf.reduce_mean(tf.square(v - v_pred)) + self.regularization()

    @tf.function
    def grad(self, X, v):
        """Compute the loss and its derivatives w.r.t. the inputs."""
        with tf.GradientTape() as tape:
            loss_value = self.loss(v, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        """Convenience method. Should be extended if needed."""
        var = self.model.trainable_variables
        return var

    def tf_optimization(self, X_v, v, tf_epochs):
        """Run the training loop."""
        for epoch in range(tf_epochs):
            loss_value = self.tf_optimization_step(X_v, v)
            self.logger.log_train_epoch(epoch, loss_value)
        return loss_value

    @tf.function
    def tf_optimization_step(self, X_v, v):
        """For each epoch, get loss+grad and backpropagate it."""
        loss_value, grads = self.grad(X_v, v)
        self.tf_optimizer.apply_gradients(
            zip(grads, self.wrap_training_variables()))
        return loss_value

    def fit(self, X_v, v, epochs, logger):
        """Train the model over a given dataset, and parameters."""
        # Setting up logger
        self.logger = logger
        self.logger.log_train_start()

        # Normalizing and preparing inputs
        X_v = self.normalize(X_v)
        # X_v = self.tensor(X_v)
        v = self.tensor(v)

        # Optimizing
        last_loss = self.tf_optimization(X_v, v, epochs)

        self.logger.log_train_end(epochs, last_loss)

    def fetch_minibatch(self, X_v, v):
        """Return a subset of the training set, for lower memory training."""
        if self.batch_size < 1:
            return self.tensor(X_v), self.tensor(v)
        N_v = X_v.shape[0]
        idx_v = np.random.choice(N_v, self.batch_size, replace=False)
        X_v_batch = self.tensor(X_v[idx_v, :])
        v_batch = self.tensor(v[idx_v, :])
        return X_v_batch, v_batch

    def predict(self, X):
        """Get the prediction for a new input X."""
        X = self.normalize(X)
        return self.model(X).numpy()

    def summary(self):
        """Print a summary of the TensorFlow/Keras model."""
        return self.model.summary()

    def tensor(self, X):
        """Convert input into a TensorFlow Tensor with the class dtype."""
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def save_to(self, model_path, params_path):
        """Save the (trained) model and params for later use."""
        with open(params_path, "wb") as f:
            pickle.dump((self.layers, self.lr, self.lam, self.lb, self.ub), f)
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
            layers, lam, lr, lb, ub = pickle.load(f)
        print(f"Loading model params from {params_path}")
        model = tf.keras.models.load_model(model_path)
        return cls(layers, lam, lr, model=model, lb=lb, ub=ub)
