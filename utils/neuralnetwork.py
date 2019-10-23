import tensorflow as tf
import numpy as np
from tqdm import tqdm


class NeuralNetwork(object):
    def __init__(self, hp, logger, ub, lb):

        layers = hp["layers"]

        # Setting up the optimizers with the hyper-parameters
        self.tf_epochs = hp["tf_epochs"]
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            decay=hp["tf_decay"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"])

        self.dtype = "float64"
        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        for width in layers[1:-1]:
            self.model.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal",
                kernel_regularizer=tf.keras.regularizers.l2(hp["lambda"])))
        self.model.add(tf.keras.layers.Dense(
                layers[-1], activation=None,
                kernel_initializer="glorot_normal",
                kernel_regularizer=tf.keras.regularizers.l2(hp["lambda"])))

        self.logger = logger
        self.ub = ub
        self.lb = lb
        self.reg_l = hp["lambda"]
        self.batch_size = hp["batch_size"]

    def normalize(self, X):
        return (X - self.lb) - 0.5*(self.ub - self.lb)

    # Defining custom loss
    @tf.function
    def loss(self, v, v_pred):
        return tf.reduce_mean(tf.square(v - v_pred))

    @tf.function
    def grad(self, X, v):
        with tf.GradientTape() as tape:
            loss_value = self.loss(v, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    def tf_optimization(self, X_v, v, tf_epochs):
        # self.logger.log_train_opt("Adam")
        for epoch in range(tf_epochs):
            # X_v_batch, u_batch = self.fetch_minibatch(X_v, v)
            loss_value = self.tf_optimization_step(X_v, v)
            self.logger.log_train_epoch(epoch, loss_value)

    @tf.function
    def tf_optimization_step(self, X_v, v):
        loss_value, grads = self.grad(X_v, v)
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def fit(self, X_v, v, custom_tf_epochs=None):
        tf_epochs = self.tf_epochs
        if custom_tf_epochs is not None:
            tf_epochs = custom_tf_epochs
        self.logger.log_train_start(self)

        # Normalizing
        X_v = self.normalize(X_v)
        X_v = self.tensor(X_v)
        v = self.tensor(v)

        # Optimizing
        self.tf_optimization(X_v, v, tf_epochs)

        self.logger.log_train_end(tf_epochs)

    def fetch_minibatch(self, X_v, v):
        if self.batch_size < 1:
            return self.tensor(X_v), self.tensor(v)
        N_v = X_v.shape[0]
        idx_v = np.random.choice(N_v, self.batch_size, replace=False)
        X_v_batch = self.tensor(X_v[idx_v, :])
        v_batch = self.tensor(v[idx_v, :])
        return X_v_batch, v_batch

    def predict(self, X):
        X = self.normalize(X)
        return self.model(X).numpy()

    def summary(self):
        return self.model.summary()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)

