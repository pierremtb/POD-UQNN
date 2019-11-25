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
        self.losses = []
        self.errors = []
        self.rel_mean_errors = []
        self.rel_std_errors = []

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

    def log_train_epoch(self, epoch, loss, error, custom="", is_iter=False):
        self.pbar.update(1)
        self.pbar.set_description(f"L: {loss:.4e}")
        if epoch % self.frequency == 0:
            rel_err = self.get_error_u()
            rel_err_str = np.array2string(rel_err, formatter={'float_kind':lambda x: "%.4f" % x})

            name = 'nt_epoch' if is_iter else '#'
            message = f"{name}: {epoch:6d} " + \
                      f"L: {loss:.4f} " + \
                      f"E_V: {rel_err_str} " + custom
            self.pbar.write(message)

            self.epochs.append(epoch)
            self.losses.append(loss)
            self.errors.append(error)
            self.rel_mean_errors.append(rel_err[0])
            self.rel_std_errors.append(rel_err[1])

    def log_train_opt(self, name):
        print(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, custom=""):
        self.pbar.close()
        print("==================")
        rel_err_str = np.array2string(self.get_error_u(),
                                      formatter={'float_kind':lambda x: "%.4f" % x})
        print(f"Training finished (epoch {epoch}): " +
              f"duration = {self.get_elapsed()}  " +
              f"rel_err_val = {rel_err_str}  " + custom)

    def get_logs(self):
        epochs = np.array(self.epochs)[:, None]
        losses = np.array(self.losses)[:, None]
        errors = np.array(self.errors)[:, None]
        rel_mean_errors = np.array(self.rel_mean_errors)[:, None]
        rel_std_errors = np.array(self.rel_std_errors)[:, None]
        header = "epoch\tloss\terror\trel_mean_error\trel_std_error"

        return (header, np.hstack((epochs, losses, errors, rel_mean_errors, rel_std_errors)))
