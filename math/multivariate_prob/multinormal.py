#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """
    def __init__(self, data):
        """
        Initialize the MultiNormal class with the given data.

        Args:
            data: numpy.ndarray of shape (d, n) containing the dataset,
                where d is the number of dimensions and n
                is the number of data points.

        Raises:
            TypeError: If data is not a numpy.ndarray.
            ValueError: If data contains fewer than 2 data points.
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")

        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Number of data points
        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean (d, 1) where d is the number of dimensions
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate the covariance matrix (d, d)
        # Number of dimensions
        d = data.shape[0]
        # Center the data around the mean
        centered_data = data - self.mean
        self.cov = (centered_data @ centered_data.T) / (n - 1)
