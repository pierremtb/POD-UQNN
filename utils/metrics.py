import numpy as np
import tensorflow as tf


def error_pod(U, V):
    return np.linalg.norm(U - V.dot(V.T).dot(U)) / np.linalg.norm(U)


def error_podnn(U, U_pred):
    err = tf.norm(U - U_pred, ord=1) / \
           tf.norm(U, ord=1)
    return err.numpy()
