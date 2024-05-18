#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    function to create mini batches
    """

    m = X.shape[0]
    mini_batches = []

    X, Y = shuffle_data(X, Y)

    num_complete_minibatches = m // batch_size

    for i in range(num_complete_minibatches):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        Y_batch = Y[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X[num_complete_minibatches * batch_size:]
        Y_batch = Y[num_complete_minibatches * batch_size:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
