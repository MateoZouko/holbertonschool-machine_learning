#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset
    """
    U, s, V = np.linalg.svd(X)
    cum = np.cumsum(s**2) / np.sum(s**2)
    k = np.argmax(cum >= var) + 1
    W = V.T[:, :k]
    return W
