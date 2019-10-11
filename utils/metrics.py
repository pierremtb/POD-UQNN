import numpy as np
import tensorflow as tf


def error_pod(U_h, V):
    return np.linalg.norm(U_h - V.dot(V.T).dot(U_h)) / np.linalg.norm(U_h)


def error_podnn(U_h, U_h_pred):
    return tf.norm(U_h - U_h_pred, ord=1) / \
           tf.norm(U_h, ord=1)
