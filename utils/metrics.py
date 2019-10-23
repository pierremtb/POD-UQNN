import numpy as np
import tensorflow as tf
from tqdm import tqdm


def error_pod(U, V):
    n_s = U.shape[1]
    err_pod = 0.0
    print("Computing POD error...")
    VV = V.dot(V.T)
    for j in tqdm(range(n_s)):
        err_pod += tf.norm(U[:, j] - VV.dot(U[:, j])) / tf.norm(U[:, j])
    return err_pod.numpy() / n_s


def error_podnn(U, U_pred):
    err = tf.norm(U - U_pred) / \
           tf.norm(U)
    return err.numpy()
