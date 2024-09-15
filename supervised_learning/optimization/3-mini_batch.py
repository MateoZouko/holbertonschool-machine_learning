#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np


def create_mini_batches(X, Y, batch_size):
    """
    function that creates mini-batches from a dataset
    """
    m = X.shape[0]
    mini_batches = []
    shuffler = np.random.permutation(m)
    X_shuffled = X[shuffler]
    Y_shuffled = Y[shuffler]
    num_batches = m // batch_size
    for i in range(num_batches):
        X_mini = X_shuffled[i * batch_size:(i + 1) * batch_size]
        Y_mini = Y_shuffled[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_mini, Y_mini))
    if m % batch_size != 0:
        X_mini = X_shuffled[num_batches * batch_size:]
        Y_mini = Y_shuffled[num_batches * batch_size:]
        mini_batches.append((X_mini, Y_mini))
    return mini_batches
