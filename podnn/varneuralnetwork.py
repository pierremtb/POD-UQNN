
"""Module with a class defining an Artificial Neural Network."""

import os
import pickle
import tensorflow as tf
import numpy as np

NORM_NONE = "none"
NORM_MEANSTD = "meanstd"
NORM_CENTER = "center"

class VarNeuralNetwork:
    def __init__(self, layers, lr, lam, adv_eps=0., norm=NORM_NONE, model=None, norm_bounds=None):
        # Making sure the dtype is consistent
        self.dtype = "float64"

        # Setting up optimizer and params
        self.tf_optimizer = tf.keras.optimizers.Adam(lr)
        self.layers = layers
        self.lr = lr
        self.lam = lam
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

        inputs = tf.keras.Input(shape=(self.layers[0],), name="x", dtype=self.dtype)

        x = inputs
        for width in self.layers[1:-1]:
            x = tf.keras.layers.Dense(
                    width, activation=tf.nn.relu, dtype=self.dtype,
                    kernel_initializer="glorot_normal")(x)
        x = tf.keras.layers.Dense(
                2 * self.layers[-1], activation=None, dtype=self.dtype,
                kernel_initializer="glorot_normal")(x)

        # Output processing function
        def split_mean_var(data):
            mean, out_var = tf.split(data, num_or_size_splits=2, axis=1)
            # var = tf.math.log(1.0 + tf.exp(out_var)) + 1e-6
            var = tf.math.softplus(out_var) + 1e-6
            return [mean, var]
        
        outputs = tf.keras.layers.Lambda(split_mean_var)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="varnn")

        return model

    def set_normalize_bounds(self, X):
        if self.norm == NORM_CENTER:
            lb = np.amin(X, axis=0)
            ub = np.amax(X, axis=0)
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

    def regularization(self):
        l2_norms = [tf.nn.l2_loss(v) for v in self.wrap_training_variables()]
        l2_norm = tf.reduce_sum(l2_norms)
        return self.lam * l2_norm
        
    @tf.function
    def loss(self, y, y_pred):
        """Return the Gaussian NLL loss function between the pred and val."""
        y_pred_mean, y_pred_var = y_pred
        return tf.reduce_mean(tf.math.log(y_pred_var) / 2) + \
               tf.reduce_mean(tf.divide(tf.square(y -  y_pred_mean), 2*y_pred_var)) + \
               self.regularization()

    @tf.function
    def grad(self, X, v):
        """Compute the loss and its derivatives w.r.t. the inputs."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            loss_value = self.loss(v, self.model(X))
            if self.adv_eps is not None:
                loss_x = tape.gradient(loss_value, X)
                X_adv = X + self.adv_eps * tf.math.sign(loss_x)
                loss_value += self.loss(v, self.model(X_adv))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        del tape
        return loss_value, grads

    def wrap_training_variables(self):
        """Convenience method. Should be extended if needed."""
        var = self.model.trainable_variables
        return var

    def tf_optimization(self, X_v, v, tf_epochs, nolog=False):
        """Run the training loop."""
        for epoch in range(tf_epochs):
            loss_value = self.tf_optimization_step(X_v, v)
            if not nolog:
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
        self.set_normalize_bounds(X_v)
        X_v = self.normalize(X_v)
        v = self.tensor(v)

        # Optimizing
        last_loss = self.tf_optimization(X_v, v, epochs)

        self.logger.log_train_end(epochs, last_loss)

    def fit_simple(self, X_v, v, epochs):
        """Train the model over a given dataset, and parameters."""
        self.set_normalize_bounds(X_v)
        X_v = self.normalize(X_v)
        v = self.tensor(v)
        # Optimizing
        self.tf_optimization(X_v, v, epochs, nolog=True)

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
        y_pred_mean, y_pred_var = self.model(X)
        return y_pred_mean.numpy(), y_pred_var.numpy()

    def summary(self):
        """Print a summary of the TensorFlow/Keras model."""
        return self.model.summary()

    def tensor(self, X):
        """Convert input into a TensorFlow Tensor with the class dtype."""
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def save_to(self, model_path, params_path):
        """Save the (trained) model and params for later use."""
        with open(params_path, "wb") as f:
            pickle.dump((self.layers, self.lr, self.lam, self.norm, self.norm_bounds), f)
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
            layers, lr, lam, norm, norm_bounds = pickle.load(f)
        print(f"Loading model params from {params_path}")
        model = tf.keras.models.load_model(model_path)
        return cls(layers, lr, lam, model=model, norm=norm, norm_bounds=norm_bounds)
