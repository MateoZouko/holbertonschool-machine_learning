#!/usr/bin/env python3
"""
Task 3
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    function that creates mini-batches from a data set
    """
    m = X.shape[0]
    mini_batches = []
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    for i in range(0, m, batch_size):
        X_mini = X_shuffled[i:i + batch_size]
        Y_mini = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_mini, Y_mini))
    return mini_batches
