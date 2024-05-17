#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle the data.
    """

    permutation = np.random.permutation(len(Y))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y
