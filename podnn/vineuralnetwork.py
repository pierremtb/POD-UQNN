
"""Module with a class defining an Artificial Neural Network."""

import os
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions

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


class DenseVariational(tfk.layers.Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = tfk.activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=tfk.initializers.RandomNormal(stddev=self.init_sigma),
                                         trainable=True, dtype=self.dtype)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tfk.initializers.RandomNormal(stddev=self.init_sigma),
                                       trainable=True, dtype=self.dtype)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=tfk.initializers.constant(0.0),
                                          trainable=True, dtype=self.dtype)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tfk.initializers.constant(0.0),
                                        trainable=True, dtype=self.dtype)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.tensor(self.kernel_rho))
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape, dtype=self.dtype)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape, dtype=self.dtype)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.tensor(self.prior_sigma_1))
        comp_2_dist = tfp.distributions.Normal(0.0, self.tensor(self.prior_sigma_2))
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))

    def tensor(self, x):
        return tf.convert_to_tensor(x, dtype=self.dtype)


class VINeuralNetwork:
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
        inputs = tfk.layers.Input((self.layers[0],), dtype=self.dtype)

        prior_params = {
            'prior_sigma_1': 1.5, 
            'prior_sigma_2': 0.1, 
            'prior_pi': 0.5 
        }

        x = inputs
        for width in self.layers[1:-1]:
            x = DenseVariational(width, self.klw, **prior_params, activation="softplus", dtype=self.dtype)(x)
        x = DenseVariational(self.layers[-1], self.klw, **prior_params, dtype=self.dtype)(x)
        # x = DenseVariational(self.layers[-1] * 2, self.klw, **prior_params)(x)

        # Output processing function
        # def split_mean_var(data):
        #     mean, out_var = tf.split(data, num_or_size_splits=2, axis=1)
        #     var = tf.math.log(1.0 + tf.exp(out_var)) + 1e-6
        #     # var = tf.math.softplus(out_var) + 1e-6
        #     var = tf.ones_like(var) * 9.
        #     return [mean, var]
        # outputs = tf.keras.layers.Lambda(split_mean_var)(x)

        outputs = x

        # outputs = tfp.layers.DistributionLambda(
        #         lambda t: tfd.Normal(loc=t[..., :self.layers[-1]],
        #                             scale=1e-3 + tf.math.softplus(0.01 * t[..., self.layers[-1]:])))(x)

        def neg_log_likelihood(y_obs, y_pred, sigma=5.0):
            dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
            return K.sum(-dist.log_prob(y_obs))

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vinn")
        model.compile(loss=neg_log_likelihood, optimizer=tfk.optimizers.Adam(self.lr))

        return model

    @tf.function
    def loss(self, y, y_pred):
        """Return the Gaussian NLL loss function between the pred and val."""
        sigma = 10.0
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        return -dist.log_prob(y)
        # y_pred_mean = y_pred[0]
        # y_pred_var = y_pred[1]
        # return tf.reduce_mean(tf.math.log(y_pred_var) / 2) + \
        #        tf.reduce_mean(tf.divide(tf.square(y -  y_pred_mean), 2*y_pred_var))

    @tf.function
    def grad(self, X, v):
        """Compute the loss and its derivatives w.r.t. the inputs."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            loss_value = self.loss(v, self.model(X))
            # if self.adv_eps is not None:
            #     loss_x = tape.gradient(loss_value, X)
            #     X_adv = X + self.adv_eps * tf.math.sign(loss_x)
            #     loss_value += self.loss(v, self.model(X_adv))
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

    def fit(self, X_v, v, epochs, logger, batch_size=32):
        """Train the model over a given dataset, and parameters."""
        # Setting up logger
        self.logger = logger
        self.logger.log_train_start()

        # Normalizing and preparing inputs
        self.set_normalize_bounds(X_v)
        X_v = self.normalize(X_v)
        v = self.tensor(v)

        # Optimizing
        self.model.fit(X_v, v, epochs=epochs, verbose=0, callbacks=[MyCallback(logger)])
        # last_loss = self.tf_optimization(X_v, v, epochs)

        # self.logger.log_train_end(epochs, last_loss)

    # def predict(self, X, samples=200):
    #     """Get the prediction for a new input X."""
    #     X = self.normalize(X)
    #     y_pred_mean, y_pred_var = self.model(X)
    #     return y_pred_mean.numpy(), y_pred_var.numpy()

    def predict(self, X, samples=200):
        """Get the prediction for a new input X."""
        X = self.normalize(X)
        yhats_mean = np.zeros((samples, X.shape[0], self.layers[-1]))
        # yhats_var = np.zeros((samples, X.shape[0], self.layers[-1]))
        for i in range(samples):
            dist = self.model(X)
            # y_pred_mean, y_pred_var = dist.mean(), dist.variance()
            y_pred_mean = self.model(X)
            yhats_mean[i] = y_pred_mean.numpy()
            # yhats_var[i] = y_pred_var.numpy()
        yhat = yhats_mean.mean(0)
        yhat_var = yhats_mean.var(0)
        # yhat_var = (yhats_var + yhats_mean ** 2).mean(0) - yhat ** 2
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
