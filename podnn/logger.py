import yaml
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm


class Logger(object):
    def __init__(self, epochs, frequency):
        # print("Hyperparameters:")
        # print(json.dumps(HP, indent=2))
        # print()

        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.tf_epochs = epochs
        self.frequency = frequency

        self.epochs = []
        self.losses = []
        self.errors = []
        self.rel_errors = []

    def get_epoch_duration(self):
        now = time.time()
        edur = datetime.fromtimestamp(now - self.prev_time) \
            .strftime("%S.%f")[:-5]
        self.prev_time = now
        return edur

    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time) \
                .strftime("%M:%S")

    def get_error_u(self):
        return np.around(self.error_fn(), decimals=4)

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model, model_description=False):
        print("\nTraining started")
        print("================")
        self.model = model
        # self.pbar = tqdm(total=self.tf_epochs)
        if model_description:
            print(model.summary())

    def log_train_epoch(self, epoch, loss, error, custom="", is_iter=False):
        # self.pbar.update(1)
        if epoch % self.frequency == 0:
            rel_error = self.get_error_u()
            name = 'nt_epoch' if is_iter else '#'
            message = f"{name}: {epoch:6d} " + \
                  f"ET: {self.get_elapsed()} " + \
                  f"L: {loss:.4e} " + \
                  f"E_V: {self.get_error_u()} " + custom
                #   f"E_V: {self.get_error_u():.4e} " + custom
                #   f"E_V: {100 * self.get_error_u():.4f}%  " + custom
            # self.pbar.set_description(f"l:{loss:.2e} e:{self.get_error_u():.2e}")
            print(message)

            self.epochs.append(epoch)
            self.losses.append(loss)
            self.errors.append(error)
            self.rel_errors.append(rel_error)

    def log_train_opt(self, name):
        print(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): " +
              f"duration = {self.get_elapsed()}  " +
              f"err_val = {100 * self.get_error_u():.4f}%  " + custom)

    def get_logs(self):
        epochs = np.array(self.epochs)[:, None]
        losses = np.array(self.losses)[:, None]
        errors = np.array(self.errors)[:, None]
        rel_errors = np.array(self.rel_errors)[:, None]
        header = "epoch\tloss\terror\trel_error"

        return (header, np.hstack((epochs, losses, errors, rel_errors)))
