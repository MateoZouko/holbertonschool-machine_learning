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
    cum = np.cumsum(s)
    W = V.T
    return W
