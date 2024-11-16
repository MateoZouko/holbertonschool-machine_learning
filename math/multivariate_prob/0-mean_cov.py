#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set where
           n is the number of data points and d is the number of dimensions.

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean of the data set
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
             of the data set.

    Raises:
        TypeError: If X is not a 2D numpy.ndarray.
        ValueError: If X contains less than 2 data points.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")

    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0).reshape(1, d)
    cov = np.dot((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
