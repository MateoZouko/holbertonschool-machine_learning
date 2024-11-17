#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset
    """
    U, s, Vt = np.linalg.svd(X)
    explained_variance = (s ** 2) / np.sum(s ** 2)
    cumulative_variance = np.cumsum(explained_variance)
    k = int(np.searchsorted(cumulative_variance, var) + 1)
    W = Vt[:k].T
    return W
