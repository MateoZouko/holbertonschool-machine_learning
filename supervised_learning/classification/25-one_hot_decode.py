#!/usr/bin/env python3
"""
Task 25
"""
import numpy as np


def one_hot_decode(Y, classes):
    """
    converts a one-hot matrix into a vector of labels:
    """
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= 0:
        return None
    if len(Y.shape) != 2:
        return None
    if Y.shape[1] != classes:
        return None
    return np.argmax(Y, axis=1)
