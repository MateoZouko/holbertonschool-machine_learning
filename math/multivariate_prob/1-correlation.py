#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing the covariance matrix.

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix.

    Raises:
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C does not have shape (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calculate the correlation matrix
    d = C.shape[0]
    # Standard deviation for each dimension
    std_dev = np.sqrt(np.diagonal(C))
    # Normalize the covariance matrix
    corr_matrix = C / (std_dev[:, None] * std_dev[None, :])

    return corr_matrix
