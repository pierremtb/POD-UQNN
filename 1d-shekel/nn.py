import tensorflow as tf
import numpy as np


class NeuralNetwork(object):
    def __init__(self, hp, logger):

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

        self.reg_l = hp["lambda"]

    # Defining custom loss
    @tf.function
    def loss(self, u, u_pred):
        return tf.reduce_mean(tf.square(u - u_pred))

    @tf.function
    def grad(self, X, u):
        with tf.GradientTape() as tape:
            loss_value = self.loss(u, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    def tf_optimization(self, X_u, u):
        self.logger.log_train_opt("Adam")
        for epoch in range(self.tf_epochs):
            loss_value = self.tf_optimization_step(X_u, u)
            self.logger.log_train_epoch(epoch, loss_value)

    @tf.function
    def tf_optimization_step(self, X_u, u):
        loss_value, grads = self.grad(X_u, u)
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def fit(self, X_u, u):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.tensor(X_u)
        u = self.tensor(u)

        # Optimizing
        self.tf_optimization(X_u, u)

        self.logger.log_train_end(self.tf_epochs)

    def predict(self, X_star):
        u_pred = self.model(X_star)
        return u_pred.numpy()

    def summary(self):
        return self.model.summary()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)

