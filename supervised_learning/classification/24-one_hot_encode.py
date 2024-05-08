#!/usr/bin/env python3
"""
Task 24
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    """

    if type(Y) is not np.ndarray or type(classes) is not int or classes < 2 \
            or classes < np.max(Y):
        return None

    n = np.zeros((classes, Y.shape[0]))
    for i, j in enumerate(Y):
        n[j, i] = 1
    return n
