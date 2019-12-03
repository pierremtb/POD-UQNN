"""Module with a class defining an Artificial Neural Network."""

import os
import pickle
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm


class NeuralNetwork:
    def __init__(self, layers, reg_lam, model=None):
        # Setting up the optimizers with the hyper-parameters
        self.dtype = "float64"
        self.tf_optimizer = None
        self.logger = None

        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)

        if model is None:
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
            for width in layers[1:-1]:
                self.model.add(tf.keras.layers.Dense(
                    width, activation=tf.nn.tanh,
                    kernel_initializer="glorot_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(reg_lam)))
            self.model.add(tf.keras.layers.Dense(
                    layers[-1], activation=None,
                    kernel_initializer="glorot_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(reg_lam)))
        else:
            self.model = model

        self.batch_size = 0

        self.layers = layers
        self.reg_lam = reg_lam
        self.lb = None
        self.ub = None

    def normalize(self, X):
        """Apply a kind of normalization to the inputs X."""
        if self.lb is not None and self.ub is not None:
            return (X - self.lb) - 0.5*(self.ub - self.lb)
        return X

    def regularization(self):
        l2_norms = [tf.nn.l2_loss(v) for v in self.wrap_training_variables()]
        l2_norm = tf.reduce_sum(l2_norms)
        return self.reg_lam * l2_norm

    # Defining custom loss
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

    @tf.function
    def tf_optimization_step(self, X_v, v):
        """For each epoch, get loss+grad and backpropagate it."""
        loss_value, grads = self.grad(X_v, v)
        self.tf_optimizer.apply_gradients(
            zip(grads, self.wrap_training_variables()))
        return loss_value

    def fit(self, X_v, v, epochs, logger, lr,
            decay=None, lb=None, ub=None):
        """Train the model over a given dataset, and parameters."""
        # Setting up logger
        self.logger = logger
        self.logger.log_train_start()

        # Setting up optimizer
        self.tf_optimizer = tf.keras.optimizers.Adam(lr, decay=decay)
        self.model.compile(self.tf_optimizer)
        self.model.summary()

        # Normalizing and preparing inputs
        self.lb = lb
        self.ub = ub
        X_v = self.normalize(X_v)
        X_v = self.tensor(X_v)
        v = self.tensor(v)

        # Optimizing
        self.tf_optimization(X_v, v, epochs)

        self.logger.log_train_end(epochs)

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
            pickle.dump((self.layers, self.reg_lam), f)
        tf.keras.models.save_model(self.model, model_path)

    @classmethod
    def load_from(cls, model_path, params_path):
        """Load a (trained) model and params."""

        if not os.path.exists(model_path):
            raise FileNotFoundError("Can't find cached model.")
        if not os.path.exists(params_path):
            raise FileNotFoundError("Can't find cached model params.")

        print(f"Loading model from {model_path}...")
        print(f"Loading model params from {params_path}...")
        with open(params_path, "rb") as f:
            layers, reg_lam = pickle.load(f)
        model = tf.keras.models.load_model(model_path)
        return cls(layers, reg_lam, model)
