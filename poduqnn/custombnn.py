"""Module with a class defining a Bayesian Neural Network."""

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
NORM_MINMAX = "minmax"

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions


class DenseVariational(tfk.layers.Layer):
    """Bayesian Inference layer adapted from
       http://krasserm.github.io/2019/03/14/bayesian-neural-networks/"""
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.0,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = tfk.activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)
        self.kernel_mu = None
        self.bias_mu = None
        self.kernel_rho = None
        self.bias_rho = None
        self.prior_mu = None
        super().__init__(**kwargs)

    def get_config(self):
        """Overriden method to allow for saving/loading."""
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'kl_weight': self.kl_weight,
            'activation': self.activation,
            'prior_sigma_1': self.prior_sigma_1,
            'prior_sigma_2': self.prior_sigma_2,
            'prior_pi': self.prior_pi,
        })
        return config

    def compute_output_shape(self, input_shape):
        """Overriden method defining sizing."""
        return input_shape[0], self.units

    def build(self, input_shape):
        """Overriden method defining the custom weights/biases."""
        k_init = tfk.initializers.RandomNormal(stddev=self.tensor(self.init_sigma))
        b_init = tfk.initializers.RandomNormal(stddev=self.tensor(self.init_sigma))
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=k_init,
                                         dtype=self.dtype,
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=b_init,
                                       dtype=self.dtype,
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=tfk.initializers.Constant(0.),
                                          dtype=self.dtype,
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=tfk.initializers.Constant(0.),
                                        dtype=self.dtype,
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """Overriden method defining the forward pass."""
        kernel_sigma = 1e-3 + tf.math.softplus(0.1 * self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma \
                 * tf.random.normal(self.kernel_mu.shape, dtype=self.dtype)
        bias_sigma = 1e-3 + tf.math.softplus(0.1 * self.bias_rho)
        bias = self.bias_mu + bias_sigma \
                 * tf.random.normal(self.bias_mu.shape, dtype=self.dtype)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))
        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        """Kullback-Leibler loss to minimize."""
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        """Prior on the weights, as a log."""
        comp_1_dist = tfp.distributions.Normal(0.0, self.tensor(self.prior_sigma_1))
        comp_2_dist = tfp.distributions.Normal(0.0, self.tensor(self.prior_sigma_2))
        c = np.log(np.expm1(1.))
        return K.log(c + self.prior_pi_1 * comp_1_dist.prob(w)
                     + self.prior_pi_2 * comp_2_dist.prob(w))

    def tensor(self, x):
        """Helper to make sure quantities are tensor of dtype."""
        return tf.convert_to_tensor(x, dtype=self.dtype)


class BayesianNeuralNetwork:
    """Custom class defining a Bayesian Neural Network model."""
    def __init__(self, layers, lr, klw,
                 soft_0=0.01, adv_eps=None,
                 pi_1=1.0, pi_2=0.1,
                 norm=NORM_NONE, model=None, norm_bounds=None):
        # Making sure the dtype is consistent
        self.dtype = "float64"
        tf.keras.backend.set_floatx(self.dtype)

        # Setting up optimizer and params
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.layers = layers
        self.lr = lr
        self.klw = klw
        self.norm_bounds = norm_bounds
        self.logger = None
        self.batch_size = 0
        self.norm = norm
        self.adv_eps = adv_eps

        self.pi_1 = pi_1
        self.pi_2 = pi_2

        # Setting up the model
        tf.keras.backend.set_floatx(self.dtype)
        if model is None:
            self.model = self.build_model()
        else:
            self.model = model

    def build_model(self):
        """Functional Keras model."""
        inputs = tf.keras.Input(shape=(self.layers[0],), name="x", dtype=self.dtype)
        x = inputs
        for width in self.layers[1:-1]:
            x = DenseVariational(
                    width, activation=tf.nn.relu, dtype=self.dtype,
                    prior_sigma_1=self.pi_1, prior_sigma_2=self.pi_2,
                    kl_weight=self.klw)(x)
        x = DenseVariational(
                2 * self.layers[-1], activation=None, dtype=self.dtype,
                kl_weight=self.klw)(x)

        # Output processing function
        def split_mean_var(data):
            mean, out_var = tf.split(data, num_or_size_splits=2, axis=1)
            # var = tf.math.log(1.0 + tf.exp(out_var)) + 1e-6
            var = tf.math.softplus(0.01 * out_var) + 1e-6
            return [mean, var]
        
        outputs = tf.keras.layers.Lambda(split_mean_var)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bnn")
        return model

    @tf.function
    def loss(self, y_obs, y_pred):
        """Negative Log-Likelihood loss function."""
        y_pred_mean, y_pred_var = y_pred
        dist = tfp.distributions.Normal(loc=y_pred_mean, scale=tf.math.sqrt(y_pred_var))
        return K.sum(-dist.log_prob(y_obs))

    @tf.function
    def grad(self, X, v):
        """Compute the loss and its derivatives w.r.t. the inputs."""
        with tf.GradientTape() as tape:
            loss_value = self.loss(v, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_trainable_variables())
        return loss_value, grads

    def wrap_trainable_variables(self):
        """Wrapper of all trainable variables."""
        return self.model.trainable_variables

    def set_normalize_bounds(self, X):
        """Setting the normalization bounds, according to the chosen method."""
        if self.norm == NORM_CENTER or self.norm == NORM_MINMAX:
            lb = X.min(0)
            ub = X.max(0)
            self.norm_bounds = (lb, ub)
        elif self.norm == NORM_MEANSTD:
            lb = X.mean(0)
            ub = X.std(0)
            self.norm_bounds = (lb, ub)

    def normalize(self, X):
        """Perform the normalization on the inputs."""
        if self.norm_bounds is None:
            return self.tensor(X)
        if self.norm == NORM_CENTER:
            lb, ub = self.norm_bounds
            X = (X - lb) - 0.5 * (ub - lb)
        elif self.norm == NORM_MEANSTD:
            mean, std = self.norm_bounds
            X = (X - mean) / std
        return self.tensor(X)

    def fit(self, X_v, v, epochs, logger=None):
        """Train the model over a given dataset, and parameters."""
        # Setting up logger
        self.logger = logger
        if self.logger is not None:
            self.logger.log_train_start()

        # Normalizing and preparing inputs
        self.set_normalize_bounds(X_v)
        X_v = self.normalize(X_v)
        v = self.tensor(v)

        # Optimizing
        for e in range(epochs):
            loss_value, grads = self.grad(X_v, v)
            self.optimizer.apply_gradients(
                zip(grads, self.wrap_trainable_variables()))
            if self.logger is not None:
                self.logger.log_train_epoch(e, loss_value)

        if self.logger is not None:
            self.logger.log_train_end(epochs, tf.constant(0., dtype=self.dtype))

    def predict_dist(self, X):
        """Get the prediction distribution for a new input X."""
        X = self.normalize(X)
        y_pred_mean, y_pred_var = self.model(X)
        dist = tfp.distributions.Normal(loc=y_pred_mean, scale=tf.math.sqrt(y_pred_var))
        return dist

    def predict(self, X, samples=100):
        """Get the prediction for a new input X."""
        X = self.normalize(X)
        v_pred_samples = np.zeros((samples, X.shape[0], self.layers[-1]))
        v_pred_var_samples = np.zeros((samples, X.shape[0], self.layers[-1]))

        for i in range(samples):
            v_pred, v_pred_var = self.model(X)
            v_pred_samples[i] = v_pred.numpy()
            v_pred_var_samples[i] = v_pred_var.numpy()

        # Approximate the mixture in a single Gaussian distribution
        v_pred = v_pred_samples.mean(0)
        v_pred_var = (v_pred_var_samples + v_pred_samples ** 2).mean(0) - v_pred ** 2
        return v_pred, v_pred_var

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
        custom_dict = {"DenseVariational": DenseVariational}
        model = tf.keras.models.load_model(model_path,
                                           custom_objects=custom_dict)
        return cls(layers, lr, klw, model=model, norm=norm, norm_bounds=norm_bounds)
