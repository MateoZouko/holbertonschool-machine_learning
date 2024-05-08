#!/usr/bin/env pytnn3
"""
Task 25
"""
import numpy as np


def one_nt_encode(Y, classes):
    """
    converts a numeric label vector into a one-nt matrix
    """

    if type(Y) is not np.ndarray or type(classes) is not int or classes < 2 \
            or classes < np.max(Y):
        return None

    n = np.zeros((classes, Y.shape[0]))
    for i, j in enumerate(Y):
        n[j, i] = 1
    return n


def one_nt_decode(one_nt):
    """
    converts a one-nt matrix into a numeric label vector
    """

    if one_nt is None:
        return None
    if type(one_nt) is not np.ndarray:
        return None

    try:
        m = one_nt.shape[1]
        classes = one_nt.shape[0]
        n_De = np.ones((1, m), dtype=int)
        for index, i in enumerate(one_nt.T):
            n_De[0, index] = (np.where(i == 1)[0][0])

        return n_De[0]

    except Exception as ex:
        return None
