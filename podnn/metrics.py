import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from tqdm import tqdm
import math


def mse(v, v_pred):
    return tf.reduce_mean(tf.square(v - v_pred))


def error_pod(U, V):
    n_s = U.shape[1]
    err_pod = 0.0
    print("Computing POD error")
    VV = V.dot(V.T)
    for j in tqdm(range(n_s)):
        err_pod += tf.norm(U[:, j] - VV.dot(U[:, j])) / tf.norm(U[:, j])
    return err_pod.numpy() / n_s


def error_podnn(U, U_pred):
    per_element_error = np.abs(U - U_pred) / np.maximum(np.abs(U), np.abs(U_pred))
    # per_element_error = np.abs(U - U_pred) / np.abs(U)
    return np.nanmean(per_element_error)
    # return norm(U - U_pred) / norm(U)

def error_podnn_tf(U, U_pred):
    return tf.reduce_mean(tf.abs(U - U_pred)/tf.abs(U))

def error_norm(U, U_pred):
    return norm(U - U_pred) / norm(U)

def error_podnn_rel(U, U_pred):
    """Define the relative error metric."""
    U_pred_mean, U_mean = np.mean(U_pred, axis=-1), np.mean(U, axis=-1)
    U_pred_std, U_std = np.std(U_pred, axis=-1), np.std(U, axis=-1)
    err_mean = error_podnn(U_mean, U_pred_mean)
    err_std = error_podnn(U_std, U_pred_std)
    return err_mean, err_std