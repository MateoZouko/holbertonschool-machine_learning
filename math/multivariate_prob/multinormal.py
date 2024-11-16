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

    def pdf(self, x):
        """
        Calculate the PDF at a data point x.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have the shape (d, 1).

        Returns:
            The value of the PDF at x.
        """
        # Validate the input x
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        # Number of dimensions
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Calculate the determinant and inverse of the covariance matrix
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)

        # Compute the quadratic form (x - mu)^T * cov_inv * (x - mu)
        diff = x - self.mean
        quadratic_form = diff.T @ cov_inv @ diff

        # Compute the PDF using the multivariate normal formula
        const = 1 / (np.sqrt((2 * np.pi) ** d * cov_det))
        pdf_value = const * np.exp(-0.5 * quadratic_form)

        # Return as scalar value
        return pdf_value.item()
