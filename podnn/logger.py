import yaml
import time
import tensorflow as tf
import numpy as np
from datetime import datetime

class LoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        self.logger = logger
    def on_epoch_end(self, epoch, logs):
        self.logger.log_train_epoch(epoch, logs["loss"])

class Logger(object):
    def __init__(self, epochs, frequency, silent=False):
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.tf_epochs = epochs
        self.frequency = frequency
        self.silent = silent

        self.epochs = []
        self.logs = []
        self.logs_keys = None
        self.get_val_err = None

        if not self.silent:
            print(f"TensorFlow version: {tf.version}")
            print(f"Eager execution: {tf.executing_eagerly()}")
            # print(f"GPU-accerelated: {len(tf.config.list_physical_devices('GPU')) > 0}")

    def get_epoch_duration(self):
        now = time.time()
        edur = datetime.fromtimestamp(now - self.prev_time) \
            .strftime("%S.%f")[:-5]
        self.prev_time = now
        return edur

    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time) \
                .strftime("%M:%S")

    def set_val_err_fn(self, fn):
        self.get_val_err = fn

    def log_train_start(self):
        if self.silent:
            return
        print("\nTraining started")
        print("================")

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if epoch % self.frequency == 0 and not self.silent:
            params = self.get_val_err() if self.get_val_err is not None else {}
            logs = {"L": loss, **params}
            if self.logs_keys is None:
                self.logs_keys = list(logs.keys())
            logs_values = [logs[x] for x in self.logs_keys]

            logs_message = ""
            for i, key in enumerate(self.logs_keys):
                if i >= 3:
                # if i >= 1:
                    logs_message += f" {key}: {logs_values[i]:.4f}"
                else:
                    logs_message += f" {key}: {logs_values[i]:.4e}"

            name = 'nt_epoch' if is_iter else '#'
            message = f"{name}: {epoch:6d} " + \
                      logs_message + " " + custom
            print(message)

    def log_train_end(self, epoch, loss, custom=""):
        if self.silent:
            return
        self.log_train_epoch(epoch, loss, custom)

        print("==================")
        print(f"Training finished (epoch {epoch}): " +
              f"duration = {self.get_elapsed()}  " + custom)

    def get_logs(self):
        if self.silent:
            return

        # epochs = np.array(self.epochs)[:, None]
        # logs = np.array(self.logs)
        # header = "epoch\t"
        # header += "\t".join(self.logs_keys)
        # return (header, np.hstack((epochs, logs)))
