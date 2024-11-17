#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset and reduces its dimensionality.

    Parameters:
    - X: numpy.ndarray of shape (n, d)
        The dataset where n is the number of
        data points and d is the number of dimensions.
    - ndim: int
        The new dimensionality of the transformed X.

    Returns:
    - T: numpy.ndarray of shape (n, ndim)
        The transformed version of X with reduced dimensions.
    """
    # Perform SVD (Singular Value Decomposition)
    U, s, Vt = np.linalg.svd(X)

    # Select the top 'ndim' components
    W = Vt[:ndim].T

    # Transform X using the weight matrix W
    T = np.dot(X, W)

    return T
