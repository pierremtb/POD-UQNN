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
    print("Computing POD error...")
    VV = V.dot(V.T)
    for j in tqdm(range(n_s)):
        err_pod += tf.norm(U[:, j] - VV.dot(U[:, j])) / tf.norm(U[:, j])
    return err_pod.numpy() / n_s


def error_podnn(U, U_pred):
    return norm(U - U_pred) / norm(U)

