#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def shuffle_data(X, Y):
    """
    function that shuffles the data points in two matrices the same way
    """
    shuffler = np.random.permutation(X.shape[0])
    return X[shuffler], Y[shuffler]
