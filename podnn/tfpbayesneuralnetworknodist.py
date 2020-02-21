
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
        inputs = tfk.layers.Input((self.layers[0],), dtype=self.dtype)

        # # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                                scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                    reinterpreted_batch_ndims=1)),
            ])
            
        # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
        def prior(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype, trainable=False),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
            ])

        x = inputs
        for width in self.layers[1:-1]:
            x = tfp.layers.DenseVariational(width, posterior_mean_field, prior,
                                          activation="relu", kl_weight=self.klw)(x)
        x = tfp.layers.DenseVariational(self.layers[-1] * 2, posterior_mean_field, prior, kl_weight=self.klw)(x)

        # Output processing function
        def split_mean_var(data):
            mean, out_var = tf.split(data, num_or_size_splits=2, axis=1)
            # var = tf.math.log(1.0 + tf.exp(out_var)) + 1e-6
            var = tf.math.softplus(out_var) + 1e-6
            return [mean, var]
        
        outputs = tf.keras.layers.Lambda(split_mean_var)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="varnn")
        return model

    @tf.function
    def loss(self, y, y_pred):
        """Return the Gaussian NLL loss function between the pred and val."""
        y_pred_mean, y_pred_var = y_pred
        return tf.reduce_mean(tf.math.log(y_pred_var) / 2) + \
               tf.reduce_mean(tf.divide(tf.square(y -  y_pred_mean), 2*y_pred_var))

    @tf.function
    def grad(self, X, v):
        """Compute the loss and its derivatives w.r.t. the inputs."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            loss_value = self.loss(v, self.model(X))
            # loss_x = tape.gradient(loss_value, X)
            # X_adv = X + self.adv_eps * tf.math.sign(loss_x)
            # loss_value += self.loss(v, self.model(X_adv))
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
        # Optimizing
        last_loss = self.tf_optimization(X_v, v, epochs)
        # self.model.fit(X_v, v, epochs=epochs, verbose=0, callbacks=[MyCallback(logger)])

        self.logger.log_train_end(epochs, last_loss)

    # def predict(self, X, samples=200):
    #     """Get the prediction for a new input X."""
    #     X = self.normalize(X)
    #     y_pred_mean, y_pred_var = self.model(X)
    #     return y_pred_mean.numpy(), y_pred_var.numpy()

    def predict(self, X, samples=200):
        """Get the prediction for a new input X."""
        X = self.normalize(X)
        yhats_mean = np.zeros((samples, X.shape[0], self.layers[-1]))
        yhats_var = np.zeros((samples, X.shape[0], self.layers[-1]))
        for i in range(samples):
            y_pred_mean, y_pred_var = self.model(X)
            yhats_mean[i] = y_pred_mean.numpy()
            yhats_var[i] = y_pred_var.numpy()
        yhat = yhats_mean.mean(0)
        yhat_var = (yhats_var + yhats_mean ** 2).mean(0) - yhat ** 2
        return yhat, np.sqrt(yhat_var)

    # def predict(self, X, samples=200):
    #     """Get the prediction for a new input X."""
    #     X = self.normalize(X)
    #     yhats_mean = np.zeros((samples, X.shape[0], self.layers[-1]))
    #     yhats_var = np.zeros((samples, X.shape[0], self.layers[-1]))
    #     for i in range(samples):
    #         dist = self.model(X)
    #         yhats_mean[i] = dist.mean().numpy()
    #         yhats_var[i] = dist.variance().numpy()
    #     yhat = yhats_mean.mean(0)
    #     yhat_var = (yhats_var + yhats_mean ** 2).mean(0) - yhat ** 2
    #     return yhat, np.sqrt(yhat_var)

    def summary(self):
        """Print a summary of the TensorFlow/Keras model."""
        return self.model.summary()

    def tensor(self, X):
        """Convert input into a TensorFlow Tensor with the class dtype."""
        return tf.convert_to_tensor(X, dtype=self.dtype)

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
