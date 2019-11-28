import yaml
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from tqdm.auto import trange, tqdm


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
        self.logs = []

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
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self):
        print("\nTraining started")
        print("================")
        # self.pbar = tqdm(total=self.tf_epochs)
        self.pbar = tqdm(total=self.tf_epochs)

    def log_train_epoch(self, epoch, logs, custom="", is_iter=False):
        self.pbar.update(1)
        self.pbar.set_description(f"L: {logs['loss']:.4e}")

        if epoch % self.frequency == 0:
            self.logs_keys = list(logs.keys())
            logs_values = [logs[x] for x in self.logs_keys]

            logs_message = ""
            for i, key in enumerate(self.logs_keys):
                logs_message += f" {key}: {logs_values[i]:.4e}"

            name = 'nt_epoch' if is_iter else '#'
            message = f"{name}: {epoch:6d} " + \
                      logs_message + custom
            self.pbar.write(message)

            self.epochs.append(epoch)
            self.logs.append(logs_values)

    def log_train_opt(self, name):
        print(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, custom=""):
        self.pbar.close()
        print("==================")
        print(f"Training finished (epoch {epoch}): " +
              f"duration = {self.get_elapsed()}  " + custom)

    def get_logs(self):
        epochs = np.array(self.epochs)[:, None]
        logs = np.array(self.logs)

        header = "epoch\t"
        header += "\t".join(self.logs_keys)
        
        return (header, np.hstack((epochs, logs)))
